"""Evidence checks must detect misleading saved outputs, not just accept valid ones."""
import copy
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from c3_verification import STRATEGIES, validate_report, verify_outputs


class EvidenceChecks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.data = Path(self.tmp.name)
        self.base = self.data / 'baseline'
        self.folder = self.base / 'comparison_budget_1_alpha_0.5'
        self.folder.mkdir(parents=True)
        pd.DataFrame({'SEQN': [1, 2, 3, 4, 5, 6],
                      'split': ['initial', 'pool', 'pool', 'validation', 'validation', 'test']}
                     ).to_csv(self.base / 'split_assignments.csv', index=False)
        pd.DataFrame({'SEQN': [1, 2, 3, 4, 5, 6], 'LUXSMED': [3, 10, 3, 3, 10, 10]}
                     ).to_csv(self.data / 'scan_outcomes.csv', index=False)
        pd.DataFrame({'SEQN': [1, 2, 3, 4, 5, 6], 'RIDAGEYR': [30] * 6}
                     ).to_csv(self.data / 'candidate_predictors.csv', index=False)
        metrics = dict(auroc=1.0, average_precision=1.0, brier_score=0.04)
        self.baseline = dict(seed=42, endpoint='Observed liver stiffness >= 8 kPa',
                             split_counts=dict(initial=1, pool=2, validation=2, test=1),
                             validation_xgboost=metrics,
                             input_sha256={n: hashlib.sha256((self.data / n).read_bytes()).hexdigest()
                                           for n in ['candidate_predictors.csv', 'scan_outcomes.csv']})
        self.report = dict(budget=1, seed=42, endpoint=self.baseline['endpoint'], hybrid_alpha=0.5,
                           baseline_validation=metrics, input_sha256=self.baseline['input_sha256'],
                           results=[dict(strategy=s, scans=1, positive_scans_found=1, precision_at_k=1.0,
                                         validation_auroc=1.0, auroc_change=0.0,
                                         validation_average_precision=1.0, validation_brier=0.04)
                                    for s in STRATEGIES])
        self.selected = pd.DataFrame([dict(strategy=s, SEQN=2, rank=1, pre_scan_score=0.8,
                                           revealed_target=1) for s in STRATEGIES])
        self.predictions = pd.DataFrame([dict(strategy=s, SEQN=pid, target=y, score=p)
                                        for s in ['Initial model', *STRATEGIES]
                                        for pid, y, p in [(4, 0, 0.2), (5, 1, 0.8)]])
        self.save_predictions()

    def save_predictions(self):
        self.predictions.to_csv(self.folder / 'validation_predictions.csv', index=False)

    def checks(self):
        return verify_outputs(self.base, self.baseline, self.report, self.selected).set_index('Check')

    def test_valid_evidence_passes(self):
        self.assertTrue(self.checks().Status.eq('Passed').all())

    def test_duplicate_selected_id_is_rejected(self):
        self.selected = pd.concat([self.selected, self.selected.iloc[:1]])
        self.assertEqual(self.checks().loc['Correct budget and unique selections', 'Status'], 'Failed')

    def test_selection_from_validation_group_is_rejected(self):
        self.selected.loc[0, 'SEQN'] = 5
        self.assertEqual(self.checks().loc['Correct budget and unique selections', 'Status'], 'Failed')

    def test_wrong_revealed_label_is_rejected(self):
        self.selected.loc[0, 'revealed_target'] = 0
        self.assertEqual(self.checks().loc['Recorded outcomes and precision', 'Status'], 'Failed')

    def test_wrong_precision_is_rejected(self):
        self.report['results'][0]['precision_at_k'] = 0.0
        self.assertEqual(self.checks().loc['Recorded outcomes and precision', 'Status'], 'Failed')

    def test_changed_input_is_rejected(self):
        with (self.data / 'candidate_predictors.csv').open('a') as handle:
            handle.write('\n')
        self.assertEqual(self.checks().loc['Original input files', 'Status'], 'Failed')

    def test_missing_predictions_are_not_a_pass(self):
        (self.folder / 'validation_predictions.csv').unlink()
        self.assertEqual(self.checks().loc['Validation metrics from saved predictions', 'Status'], 'Not available')

    def test_changed_prediction_invalidates_metrics(self):
        self.predictions.loc[2, 'score'] = 0.9
        self.save_predictions()
        self.assertEqual(self.checks().loc['Validation metrics from saved predictions', 'Status'], 'Failed')

    def test_test_participant_in_predictions_is_rejected(self):
        self.predictions.loc[2, 'SEQN'] = 6
        self.save_predictions()
        self.assertEqual(self.checks().loc['Validation metrics from saved predictions', 'Status'], 'Failed')

    def test_duplicate_partition_id_is_rejected(self):
        path = self.base / 'split_assignments.csv'
        frame = pd.read_csv(path)
        frame.loc[5, 'SEQN'] = 4
        frame.to_csv(path, index=False)
        self.assertEqual(self.checks().loc['Separate data groups', 'Status'], 'Failed')

    def test_valid_report_renders(self):
        self.assertEqual(len(validate_report(self.report, self.baseline, 1)), 5)

    def test_wrong_configuration_is_rejected_before_rendering(self):
        for key, value in [('seed', 1), ('endpoint', 'Observed liver stiffness >= 10 kPa'),
                           ('hybrid_alpha', 0.7), ('budget', 2)]:
            report = copy.deepcopy(self.report)
            report[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_report(report, self.baseline, 1)

    def test_misleading_report_values_are_rejected_before_rendering(self):
        for key, value in [('scans', 2), ('precision_at_k', 0.5), ('validation_auroc', float('nan')),
                           ('auroc_change', 0.5), ('positive_scans_found', 0.5)]:
            report = copy.deepcopy(self.report)
            report['results'][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_report(report, self.baseline, 1)


if __name__ == '__main__':
    unittest.main()
