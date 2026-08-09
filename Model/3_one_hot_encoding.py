import csv
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_FILE = SCRIPT_DIR / "1_remove_missing_rows.csv"
OUTPUT_FILE = SCRIPT_DIR / "3_one_hot_encoding.csv"
CATEGORICAL_COLUMNS = ("Site", "Energy")


def load_data(file_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not file_path.is_file():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    with file_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = reader.fieldnames

        if not fieldnames:
            raise ValueError(f"CSV has no header: {file_path}")

        missing_columns = set(CATEGORICAL_COLUMNS).difference(fieldnames)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"CSV is missing required columns: {missing}")

        return fieldnames, list(reader)


# Load Model/1_remove_missing_rows.csv into global variables.
FIELDNAMES, DATA = load_data(INPUT_FILE)


def get_categories(column_name: str) -> list[str]:
    return sorted({row[column_name] for row in DATA})


def one_hot_encode() -> tuple[list[str], list[dict[str, object]]]:
    categories = {
        column: get_categories(column) for column in CATEGORICAL_COLUMNS
    }

    output_fieldnames: list[str] = []
    for fieldname in FIELDNAMES:
        if fieldname in CATEGORICAL_COLUMNS:
            output_fieldnames.extend(
                f"{fieldname}_{category}"
                for category in categories[fieldname]
            )
        else:
            output_fieldnames.append(fieldname)

    encoded_rows: list[dict[str, object]] = []
    for row in DATA:
        encoded_row: dict[str, object] = {}

        for fieldname in FIELDNAMES:
            if fieldname in CATEGORICAL_COLUMNS:
                for category in categories[fieldname]:
                    encoded_row[f"{fieldname}_{category}"] = int(
                        row[fieldname] == category
                    )
            else:
                encoded_row[fieldname] = row[fieldname]

        encoded_rows.append(encoded_row)

    return output_fieldnames, encoded_rows


def main() -> None:
    output_fieldnames, encoded_rows = one_hot_encode()

    with OUTPUT_FILE.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=output_fieldnames)
        writer.writeheader()
        writer.writerows(encoded_rows)

    print(
        f"Saved: {OUTPUT_FILE} "
        f"({len(encoded_rows)} rows, {len(output_fieldnames)} columns)"
    )


if __name__ == "__main__":
    main()
