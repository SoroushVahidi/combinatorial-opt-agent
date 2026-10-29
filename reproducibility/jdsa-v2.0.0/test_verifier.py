#!/usr/bin/env python3
"""Failure-detection tests. All mutations occur in disposable copies."""
import importlib.util,json,shutil,sys,tempfile,unittest
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('release_verifier',ROOT/'verify.py')
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
class EvidenceIntegrity(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.copy=Path(self.tmp.name)/'release';shutil.copytree(ROOT,self.copy);v.ROOT=self.copy;v.CHECKS.clear()
    def tearDown(self):self.tmp.cleanup();v.ROOT=ROOT;v.CHECKS.clear()
    def test_intact(self):v.integrity()
    def test_modified_input_fails(self):
        p=self.copy/'data/g3_counts.json';p.write_bytes(p.read_bytes()+b' ')
        with self.assertRaises(AssertionError):v.integrity()
    def test_missing_input_fails(self):
        (self.copy/'data/g3_counts.json').unlink()
        with self.assertRaises(AssertionError):v.integrity()
    def test_unmanifested_input_fails(self):
        (self.copy/'unexpected.txt').write_text('unexpected')
        with self.assertRaises(AssertionError):v.integrity()
    def test_changed_manifest_fails_checksums(self):
        p=self.copy/'release_manifest.json';p.write_bytes(p.read_bytes()+b' ')
        with self.assertRaises(AssertionError):v.integrity()
    def test_statistical_edge_cases(self):
        self.assertEqual(v.swap_p([]),1);self.assertEqual(v.swap_p([1,-1]),1)
        self.assertEqual(v.swap_p([1,1,1]),.25)
        self.assertEqual(v.holm([.01,.04,.03]),[.03,.06,.06])
if __name__=='__main__':unittest.main()
