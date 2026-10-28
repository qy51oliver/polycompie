import io
import unittest

try:
    import openpyxl
except ModuleNotFoundError:
    openpyxl = None

from app.export import samples_to_excel


@unittest.skipIf(openpyxl is None, "Excel export app extra is not installed")
class ExcelExportTests(unittest.TestCase):
    def test_untrusted_text_round_trips_as_text_not_formulas(self):
        samples = [{
            "sample_id": 6,
            "_source_pdf": "=1+1.pdf",
            "matrix_name": [{"name": "=1+1", "description": "#N/A"}],
            "properties": {
                "=untrusted_key": {"value": "=2+2", "unit": "=unit"},
                "melting_temperature": {"value": 278.1, "unit": "°C"},
            },
        }]
        workbook = openpyxl.load_workbook(io.BytesIO(samples_to_excel(samples)))
        sheet = workbook.active
        cells = [cell for row in sheet for cell in row if cell.value is not None]
        values = [cell.value for cell in cells]
        for expected in ("=1+1.pdf", "=1+1", "#N/A", "=2+2", "=unit",
                         "=untrusted_key_value", 278.1):
            self.assertIn(expected, values)
        self.assertFalse(any(cell.data_type == "f" for cell in cells))
        for cell in cells:
            if isinstance(cell.value, str):
                self.assertEqual(cell.data_type, "s")
            elif cell.value == 278.1:
                self.assertEqual(cell.data_type, "n")
