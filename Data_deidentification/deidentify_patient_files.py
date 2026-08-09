from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tkinter import Tk, filedialog

try:
    from PyPDF2 import PdfReader, PdfWriter
    from PyPDF2.generic import ContentStream, NameObject, TextStringObject
except ImportError:  # pragma: no cover - shown to users at runtime
    PdfReader = None

try:
    import pydicom
except ImportError:  # pragma: no cover - shown to users at runtime
    pydicom = None


ALLOWED_DICOM_PATIENT_TAGS = {
    0x00100010,  # Patient Name
    0x00100020,  # Patient ID
    0x00100030,  # Patient Birth Date
    0x00100040,  # Patient Sex
}
OTHER_DICOM_PATIENT_TAGS = {
    0x00101000,  # Other Patient IDs
    0x00101001,  # Other Patient Names
}

PDF_INITIAL_DIR = Path.home()
DCM_INITIAL_DIR = Path.home()


def existing_initial_dir(path: Path) -> str | None:
    return str(path) if path.exists() and path.is_dir() else None


@dataclass
class PatientIdentity:
    original_id: str
    original_name: str
    mock_id: str
    mock_name: str


class NameMapper:
    def __init__(self, run_time: datetime):
        self.base_name = time_words(run_time)
        self.by_patient_id: dict[str, str] = {}

    def mock_name_for(self, patient_id: str) -> str:
        key = patient_id.strip()
        if key not in self.by_patient_id:
            index = len(self.by_patient_id)
            self.by_patient_id[key] = self.base_name if index == 0 else f"{self.base_name}_{index}"
        return self.by_patient_id[key]


def number_word(n: int) -> str:
    ones = [
        "ZERO",
        "ONE",
        "TWO",
        "THREE",
        "FOUR",
        "FIVE",
        "SIX",
        "SEVEN",
        "EIGHT",
        "NINE",
        "TEN",
        "ELEVEN",
        "TWELVE",
        "THIRTEEN",
        "FOURTEEN",
        "FIFTEEN",
        "SIXTEEN",
        "SEVENTEEN",
        "EIGHTEEN",
        "NINETEEN",
    ]
    tens = ["", "", "TWENTY", "THIRTY", "FORTY", "FIFTY"]
    if n < 20:
        return ones[n]
    return tens[n // 10] + (ones[n % 10] if n % 10 else "")


def time_words(run_time: datetime) -> str:
    return f"{number_word(run_time.hour)}, {number_word(run_time.minute)} {number_word(run_time.second)}"


def shift_patient_id(patient_id: str) -> str:
    result = []
    for char in patient_id.strip():
        if "A" <= char <= "Z":
            result.append(chr(ord("A") + ((ord(char) - ord("A") + 1) % 26)))
        elif "a" <= char <= "z":
            result.append(chr(ord("a") + ((ord(char) - ord("a") + 1) % 26)))
        elif "0" <= char <= "9":
            result.append(str((int(char) + 1) % 10))
        else:
            result.append(char)
    return "".join(result)


def sanitize_filename_part(value: str) -> str:
    value = value.strip().replace("\\", "_").replace("/", "_").replace(":", "_")
    value = re.sub(r'[<>:"|?*\x00-\x1f]', "_", value)
    value = re.sub(r"\s+", "_", value)
    return value or "UNKNOWN"


def build_identity(patient_id: str, patient_name: str, mapper: NameMapper) -> PatientIdentity:
    clean_id = patient_id.strip()
    clean_name = patient_name.strip()
    return PatientIdentity(
        original_id=clean_id,
        original_name=clean_name,
        mock_id=shift_patient_id(clean_id),
        mock_name=mapper.mock_name_for(clean_id),
    )


def dicom_dataset_snapshot(ds, allowed_tags: set[int] | None = None) -> list[tuple[tuple, str, str]]:
    snapshot = []
    allowed_tags = allowed_tags if allowed_tags is not None else ALLOWED_DICOM_PATIENT_TAGS

    def walk(dataset, path: tuple) -> None:
        for elem in dataset:
            tag = int(elem.tag)
            if tag in allowed_tags:
                continue
            elem_path = path + (tag,)
            if elem.VR == "SQ":
                snapshot.append((elem_path, elem.VR, str(len(elem.value))))
                for index, item in enumerate(elem.value):
                    walk(item, elem_path + (index,))
            else:
                snapshot.append((elem_path, elem.VR, repr(elem.value)))

    walk(ds, ())
    return snapshot


def assert_only_allowed_dicom_fields_changed(
    original_snapshot: list[tuple],
    output_path: Path,
    allowed_tags: set[int] | None = None,
) -> None:
    saved_ds = pydicom.dcmread(str(output_path), force=True)
    saved_snapshot = dicom_dataset_snapshot(saved_ds, allowed_tags)
    if saved_snapshot != original_snapshot:
        raise RuntimeError(
            "DICOM verification failed: a field outside Patient Name, Patient ID, "
            "Patient Birth Date, Patient Sex, Other Patient IDs, or Other Patient Names changed"
        )


def rewrite_dicom(path: Path, identity: PatientIdentity, birth_date: str, output_dir: Path) -> Path:
    if pydicom is None:
        raise RuntimeError("pydicom is required. Install it with: pip install pydicom")

    ds = pydicom.dcmread(str(path), force=True, stop_before_pixels=True)
    rt_plan_name = str(getattr(ds, "RTPlanName", "") or "RTPLAN").strip()
    file_name = f"{sanitize_filename_part(identity.mock_id)}_{sanitize_filename_part(rt_plan_name)}_MOCK.dcm"
    output_path = output_dir / file_name
    return rewrite_dicom_patient_fields(path, identity, birth_date, output_path)


def parse_pdf_identity(text: str) -> tuple[str, str]:
    match = re.search(r"Patient Name:\s*(.*?)\s*,?\s*Patient ID#:\s*([^\s|]+)", text, re.IGNORECASE | re.DOTALL)
    if not match:
        raise ValueError("Cannot find Patient Name / Patient ID# in PDF text")
    return match.group(2).strip(), " ".join(match.group(1).split()).strip(" ,")


def string_operands(operands) -> list[str]:
    values = []
    for operand in operands:
        if isinstance(operand, str):
            values.append(operand)
        elif isinstance(operand, list):
            values.extend(item for item in operand if isinstance(item, str))
    return values


def fit_pdf_text_replacement(value: str, width: int) -> str:
    if len(value) > width:
        return value[:width]
    return value.ljust(width)


def replace_substring_in_pdf_array(
    array, old_value: str, new_value: str, preserve_width: bool = True
) -> bool:
    text_parts = []
    char_locations = []
    for item_index, item in enumerate(array):
        if not isinstance(item, str):
            continue
        text = str(item)
        text_parts.append(text)
        char_locations.extend((item_index, char_index) for char_index in range(len(text)))

    joined = "".join(text_parts)
    start = joined.find(old_value)
    if start < 0:
        return False

    end = start + len(old_value)
    if not preserve_width:
        for item_index, item in enumerate(array):
            if isinstance(item, str) and old_value in str(item):
                array[item_index] = TextStringObject(str(item).replace(old_value, new_value))
                return True

        replacement_text = joined[:start] + new_value + joined[end:]
        first_text_index = char_locations[0][0]
        array[first_text_index] = TextStringObject(replacement_text)
        for item_index, item in enumerate(array):
            if item_index != first_text_index and isinstance(item, str):
                array[item_index] = TextStringObject("")
        return True

    replacement = fit_pdf_text_replacement(new_value, len(old_value))
    edited: dict[int, list[str]] = {}
    for replacement_index, joined_index in enumerate(range(start, end)):
        item_index, char_index = char_locations[joined_index]
        if item_index not in edited:
            edited[item_index] = list(str(array[item_index]))
        edited[item_index][char_index] = replacement[replacement_index]

    for item_index, chars in edited.items():
        array[item_index] = TextStringObject("".join(chars))
    return True


def replace_pdf_patient_values(page, reader: PdfReader, identity: PatientIdentity) -> None:
    content = ContentStream(page.get_contents(), reader)
    for operands, operator in content.operations:
        text = "".join(string_operands(operands))
        if operator == b"TJ":
            if not operands or not isinstance(operands[0], list):
                continue
            if "Patient Name" in text:
                replace_substring_in_pdf_array(
                    operands[0],
                    identity.original_name,
                    identity.mock_name,
                    preserve_width=False,
                )
            if "Patient ID" in text:
                replace_substring_in_pdf_array(operands[0], identity.original_id, identity.mock_id)
        elif operator in {b"Tj", b"'", b'"'}:
            if not operands or not isinstance(operands[-1], str):
                continue
            new_text = str(operands[-1])
            if "Patient Name" in text:
                new_text = new_text.replace(identity.original_name, identity.mock_name)
            if "Patient ID" in text:
                new_text = new_text.replace(
                    identity.original_id,
                    fit_pdf_text_replacement(identity.mock_id, len(identity.original_id)),
                )
            operands[-1] = TextStringObject(new_text)

    page[NameObject("/Contents")] = content


def rewrite_pdf(path: Path, mapper: NameMapper, output_dir: Path) -> Path:
    if PdfReader is None:
        raise RuntimeError("PyPDF2 is required. Install it with: pip install PyPDF2")

    reader = PdfReader(str(path))
    target_text = None
    target_page = None
    for page in reader.pages:
        text = page.extract_text() or ""
        if "Monitor Unit Calc Sheet" in text:
            target_text = text
            target_page = page
            break
    if target_text is None:
        raise ValueError(f"No page containing 'Monitor Unit Calc Sheet' found in {path}")

    patient_id, patient_name = parse_pdf_identity(target_text)
    identity = build_identity(patient_id, patient_name, mapper)
    assert target_page is not None
    replace_pdf_patient_values(target_page, reader, identity)

    output_path = output_dir / f"{sanitize_filename_part(identity.mock_id)}_MOCK.pdf"
    writer = PdfWriter()
    writer.add_page(target_page)
    with output_path.open("wb") as output_file:
        writer.write(output_file)
    return output_path


def select_files(
    title: str,
    filetypes: list[tuple[str, str]],
    initial_dir: Path | None = None,
) -> list[Path]:
    dialog_options = {"title": title, "filetypes": filetypes}
    if initial_dir is not None:
        initial_dir_value = existing_initial_dir(initial_dir)
        if initial_dir_value:
            dialog_options["initialdir"] = initial_dir_value
    paths = filedialog.askopenfilenames(**dialog_options)
    return [Path(path) for path in paths]


def is_structure_set_file(path: Path) -> bool:
    return path.suffix.lower() == ".dcm" and "strctrsets" in path.name.lower()


def structure_set_output_name(path: Path, identity: PatientIdentity) -> str:
    stem = path.stem
    original_id = identity.original_id
    mock_id = sanitize_filename_part(identity.mock_id)
    if original_id and stem.startswith(original_id):
        stem = f"{mock_id}{stem[len(original_id):]}"
    elif original_id and original_id in stem:
        stem = stem.replace(original_id, mock_id, 1)
    else:
        stem = f"{mock_id}_{stem}"
    return f"{sanitize_filename_part(stem)}_MOCK{path.suffix}"


def rewrite_structure_set_dicom(
    path: Path,
    mapper: NameMapper,
    birth_date: str,
    output_dir: Path,
) -> Path:
    patient_id, patient_name, _ = read_dicom_identity(path)
    identity = build_identity(patient_id, patient_name, mapper)
    output_path = output_dir / structure_set_output_name(path, identity)
    return rewrite_dicom_patient_fields(
        path,
        identity,
        birth_date,
        output_path,
        sync_other_patient_fields=True,
    )


def should_sync_other_patient_fields(ds, identity: PatientIdentity) -> bool:
    other_patient_ids = str(getattr(ds, "OtherPatientIDs", "")).strip()
    other_patient_names = str(getattr(ds, "OtherPatientNames", "")).strip()
    return other_patient_ids == identity.original_id and other_patient_names == identity.original_name


def rewrite_dicom_patient_fields(
    path: Path,
    identity: PatientIdentity,
    birth_date: str,
    output_path: Path,
    sync_other_patient_fields: bool = False,
) -> Path:
    if pydicom is None:
        raise RuntimeError("pydicom is required. Install it with: pip install pydicom")

    ds = pydicom.dcmread(str(path), force=True)
    allowed_tags = set(ALLOWED_DICOM_PATIENT_TAGS)
    sync_other_patient_fields = sync_other_patient_fields and should_sync_other_patient_fields(ds, identity)
    if sync_other_patient_fields:
        allowed_tags.update(OTHER_DICOM_PATIENT_TAGS)
    original_snapshot = dicom_dataset_snapshot(ds, allowed_tags)
    ds.PatientName = identity.mock_name
    ds.PatientID = identity.mock_id
    ds.PatientBirthDate = birth_date
    ds.PatientSex = "X"
    if sync_other_patient_fields:
        ds.OtherPatientIDs = identity.mock_id
        ds.OtherPatientNames = identity.mock_name
    ds.save_as(str(output_path), write_like_original=True)
    assert_only_allowed_dicom_fields_changed(original_snapshot, output_path, allowed_tags)
    return output_path


def _print_intro() -> None:
    print("=" * 64)
    print("  Data De-identification  -  Patient File Mocking Tool")
    print("=" * 64)
    print()
    print("  This tool creates de-identified mock copies of selected")
    print("  DICOM Structure Set, DICOM RTPLAN, and RadCalc PDF files.")
    print("  Original files are not modified.")
    print()
    print("  You will be asked to select three groups of files in order:")
    print("    1. DICOM Structure Set file  (*.dcm)")
    print("    2. DICOM RTPLAN files        (*.dcm)")
    print("    3. RadCalc PDF reports       (*.pdf)")
    print()
    print("  Output files are written to the 'Mock' folder next to this")
    print("  program. Patient IDs are shifted by one character/digit,")
    print("  and patient names are replaced with time-based mock names.")
    print("  Structure Set DICOM edits are strictly limited to Patient")
    print("  Name, Patient ID, Patient Birth Date, Patient Sex, and")
    print("  matching Other Patient IDs / Other Patient Names when present.")
    print()
    print("  WARNING:")
    print("    PDF outputs use only the mock patient ID in the file name.")
    print("    Files with the same ID but different sites or dates will")
    print("    overwrite each other.")
    print()
    print("  Please verify the generated files before sharing them.")
    print("=" * 64)
    print()


def main() -> int:
    _print_intro()
    script_dir = Path(__file__).resolve().parent
    output_dir = script_dir / "Mock"
    output_dir.mkdir(exist_ok=True)
    run_time = datetime.now()
    mapper = NameMapper(run_time)
    birth_date = run_time.strftime("%Y%m%d")

    root = Tk()
    root.withdraw()
    root.update()

    structure_set_files = select_files(
        "Select StrctrSets .dcm file",
        [("DICOM files", "*.dcm"), ("All files", "*.*")],
        DCM_INITIAL_DIR,
    )
    ignored_non_structure_set_files = [
        path for path in structure_set_files if not is_structure_set_file(path)
    ]
    structure_set_files = [path for path in structure_set_files if is_structure_set_file(path)]
    dcm_files = select_files(
        "Select .dcm file(s)",
        [("DICOM files", "*.dcm"), ("All files", "*.*")],
        DCM_INITIAL_DIR,
    )
    ignored_structure_set_files = [path for path in dcm_files if is_structure_set_file(path)]
    dcm_files = [path for path in dcm_files if not is_structure_set_file(path)]
    pdf_files = select_files(
        "Select .pdf file(s)",
        [("PDF files", "*.pdf"), ("All files", "*.*")],
        PDF_INITIAL_DIR,
    )

    outputs: list[Path] = []
    errors: list[str] = []
    errors.extend(
        f"{path}: ignored because non-Structure Set DICOM files must be selected in the second window"
        for path in ignored_non_structure_set_files
    )
    errors.extend(
        f"{path}: ignored because Structure Set DICOM files must be selected in the first window"
        for path in ignored_structure_set_files
    )

    for path in structure_set_files:
        try:
            outputs.append(rewrite_structure_set_dicom(path, mapper, birth_date, output_dir))
        except Exception as exc:
            errors.append(f"{path}: {exc}")

    for path in dcm_files:
        try:
            patient_id, patient_name, rt_plan_name = read_dicom_identity(path)
            identity = build_identity(patient_id, patient_name, mapper)
            outputs.append(rewrite_dicom(path, identity, birth_date, output_dir))
        except Exception as exc:
            errors.append(f"{path}: {exc}")

    for path in pdf_files:
        try:
            outputs.append(rewrite_pdf(path, mapper, output_dir))
        except Exception as exc:
            errors.append(f"{path}: {exc}")

    summary = [f"Created {len(outputs)} file(s):"]
    summary.extend(str(path) for path in outputs)
    if errors:
        summary.append("\nErrors:")
        summary.extend(errors)
    message = "\n".join(summary)
    print(message)
    if errors:
        return 1
    return 0


def read_dicom_identity(path: Path) -> tuple[str, str, str]:
    if pydicom is None:
        raise RuntimeError("pydicom is required. Install it with: pip install pydicom")

    ds = pydicom.dcmread(str(path), force=True, stop_before_pixels=True)
    patient_name = str(getattr(ds, "PatientName", "")).strip()
    patient_id = str(getattr(ds, "PatientID", "")).strip()
    rt_plan_name = str(getattr(ds, "RTPlanName", "")).strip()

    if not patient_id or not patient_name:
        raise ValueError("Cannot find DICOM Patient ID / Patient Name")
    return patient_id, patient_name, rt_plan_name


if __name__ == "__main__":
    sys.exit(main())
