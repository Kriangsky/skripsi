import hashlib
from pathlib import Path
import re
import unittest


ROOT=Path(__file__).resolve().parent
PAPER=ROOT/'Effect of Vocabulary Expansion on YOLO26 Classification for BISINDO Phrase Recognition.md'


class ProtocolAddendumTests(unittest.TestCase):
    def test_parent_and_addendum_integrity(self):
        for name in ['PREREGISTRATION','PREREGISTRATION_ADDENDUM_01']:
            expected=(ROOT/f'{name}.sha256').read_text().split()[0]
            self.assertEqual(hashlib.sha256((ROOT/f'{name}.md').read_bytes()).hexdigest(),expected)
        parent=(ROOT/'PREREGISTRATION.sha256').read_text().split()[0]
        self.assertIn(parent,(ROOT/'PREREGISTRATION_ADDENDUM_01.md').read_text())

    def test_paper_scope_and_unavailable_confirmatory_results(self):
        text=PAPER.read_text(encoding='utf8')
        self.assertIsNone(re.search(r'outperform|superior|real[- ]time',text,re.I))
        for section in ['Abstract','Introduction','Related Work','Dataset and Protocol','Method','Experiments','Results','Analysis','Limitations','Conclusion']:
            self.assertIn('## '+section,text)
        self.assertIn('PILOT EKSPLORATIF',text)
        self.assertIn('HASIL KONFIRMATORI — BELUM ADA',text)
        self.assertIn('inference cost',text)
        self.assertIn('four participants excluded from training',text)

    def test_no_unauthorized_approval_file_or_addendum02_generated(self):
        self.assertFalse((ROOT/'PREREGISTRATION_ADDENDUM_02.md').exists())
        self.assertNotIn('Final H1 evidence-domain decision: external',PAPER.read_text(encoding='utf8'))


if __name__=='__main__':unittest.main()
