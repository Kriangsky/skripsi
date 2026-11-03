import copy
import json
from pathlib import Path
import tempfile
import unittest
import shutil
import subprocess

import cv2
import numpy as np
import pandas as pd

import pilot_study as p


def inventory():
    return pd.DataFrame([dict(sample_id=f'{cls}-{i}', video=f'train/{cls}/{i}.mov',
        source='train', split='train' if i < 24 else 'val', class_name=cls,
        class_id=n, signer=f'person-{i % 4}', video_sha256=f'{n:02x}{i:062x}')
        for n, cls in enumerate(p.CLASSES) for i in range(26)])


class PilotTests(unittest.TestCase):
    def test_sampling_reproducible_stratified_train_only_diverse(self):
        data = inventory()
        sampled = p.sample_videos(data)
        self.assertEqual(len(sampled), 160)
        self.assertTrue(sampled.groupby('class_name').size().eq(20).all())
        self.assertTrue(sampled.split.eq('train').all())
        self.assertTrue(sampled.groupby('class_name').signer.nunique().eq(4).all())
        self.assertEqual(sampled.sample_id.tolist(), p.sample_videos(data.sample(frac=1)).sample_id.tolist())
        external = data.assign(source='test', split='test',
            sample_id=lambda d: 'external-' + d.sample_id,
            video=lambda d: d.video.str.replace('train/', 'test/', regex=False),
            video_sha256=lambda d: 'ff' + d.video_sha256.str[2:])
        external['video_sha256'] = [f'ff{i:062x}' for i in range(len(external))]
        with_external = p.sample_videos(pd.concat([data, external], ignore_index=True))
        self.assertEqual(sampled.sample_id.tolist(), with_external.sample_id.tolist())

    def test_sampling_missing_signers_and_insufficient_pool(self):
        data = inventory().assign(signer='')
        self.assertEqual(len(p.sample_videos(data)), 160)
        with self.assertRaises(ValueError):
            p.sample_videos(data[data.class_name.ne('tidur')])

    def test_duplicate_content_and_invalid_mapping_rejected(self):
        data = inventory()
        data.loc[1, 'video_sha256'] = data.loc[0, 'video_sha256']
        with self.assertRaises(ValueError):
            p.sample_videos(data)
        data = inventory()
        data.loc[0, 'class_id'] = 7
        with self.assertRaises(ValueError):
            p.sample_videos(data)

    def test_selection_is_class_blind_and_retains_failure(self):
        candidates = [dict(candidate_id='b', frame_index=2, focus=100., quality_flags=['blur'],
                           quality_status='flagged', readable=True),
                      dict(candidate_id='a', frame_index=1, focus=70., quality_flags=[],
                           quality_status='eligible', readable=True)]
        self.assertEqual(p.choose_candidate(candidates)['candidate_id'], 'a')
        self.assertEqual(p.choose_candidate(candidates[:1])['candidate_id'], 'b')
        self.assertIsNone(p.choose_candidate([dict(candidates[0], readable=False)]))
        self.assertEqual(p.choose_candidate(list(reversed(candidates)))['candidate_id'], 'a')

    def payload(self, reviewer='reviewer_1'):
        return dict(schema=1, study_id='study', reviewer_id=reviewer, responses=[
            dict(sample_id='opaque1', decision='saya', reasons=[], timestamp='2026-11-01T01:00:00Z'),
            dict(sample_id='opaque2', decision='ambiguous', reasons=['temporal_insufficient'], timestamp='2026-11-01T01:00:00Z')])

    def test_response_validation_rejects_stale_duplicate_incomplete_wrong_reviewer(self):
        payload = self.payload()
        p.validate_responses(payload, 'study', 'reviewer_1', {'opaque1', 'opaque2'})
        for mutate in [lambda x: x.update(study_id='stale'),
                       lambda x: x.update(reviewer_id='reviewer_2'),
                       lambda x: x['responses'].pop(),
                       lambda x: x['responses'].append(x['responses'][0]),
                       lambda x: x['responses'][0].update(decision='unknown'),
                       lambda x: x['responses'][0].update(timestamp='yesterday'),
                       lambda x: x['responses'][0].update(reasons=['unknown'])]:
            broken = copy.deepcopy(payload)
            mutate(broken)
            with self.assertRaises(ValueError):
                p.validate_responses(broken, 'study', 'reviewer_1', {'opaque1', 'opaque2'})

    def test_blind_html_contains_only_public_payload(self):
        page = p.review_html('study', 'reviewer_1', [dict(sample_id='opaque1', image='data:image/jpeg;base64,AA==')])
        for secret in ['class_name', 'video_sha256', 'candidate_id', 'train/', 'candidate_frames/', 'ground_truth']:
            self.assertNotIn(secret, page)
        self.assertIn('opaque1', page)
        self.assertNotIn('reviewer_2', page)
        self.assertIn('ambiguous', page)

    def test_metrics_full_denominator_subset_kappa_and_failures(self):
        manifest = pd.DataFrame([dict(sample_id='opaque1', class_name='saya'),
                                 dict(sample_id='opaque2', class_name='kamu')])
        selection = pd.DataFrame([dict(sample_id='opaque1', selection_status='eligible'),
                                  dict(sample_id='opaque2', selection_status='missing')])
        a = self.payload()['responses']
        b = copy.deepcopy(a)
        b[1]['decision'] = 'invalid'
        summary, per_class, agreement, matrices = p.calculate_metrics(manifest, selection, [a, b])
        self.assertEqual(summary[0]['accuracy_all'], .5)
        self.assertEqual(summary[0]['accuracy_identifiable'], 1.)
        self.assertEqual(summary[0]['identifiable_n'], 1)
        self.assertEqual(summary[0]['technical_coverage'], .5)
        self.assertEqual(agreement['agreement'], .5)
        self.assertAlmostEqual(agreement['cohens_kappa'], 1 / 3)
        self.assertEqual(sum(map(sum, matrices[0])), 2)
        self.assertEqual(sum(r['n'] for r in per_class if r['reviewer_id'] == 'reviewer_1'), 2)
        self.assertEqual(summary[0]['semantic_failure_readable_n'], 0)

    def test_pending_analysis_never_fabricates_review_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            (work / 'responses').mkdir()
            pd.DataFrame([dict(sample_id='opaque1', class_name='saya')]).to_csv(work / 'pilot_manifest_private.csv', index=False)
            pd.DataFrame([dict(sample_id='opaque1', selection_status='missing')]).to_csv(work / 'pilot_selection_report.csv', index=False)
            p.analyze(work)
            self.assertIn('PENDING', (work / 'analysis/pilot_report.md').read_text())
            self.assertFalse((work / 'analysis/confusion_matrix.png').exists())

    def fixture(self, root):
        source = root / 'v3'
        source.mkdir()
        (source / 'candidate_records').mkdir()
        (source / 'candidate_frames').mkdir()
        data = inventory()
        data['sample_id'] = [f'{i:020x}' for i in range(len(data))]
        data['video_sha256'] = ''
        candidates = []
        image = np.random.default_rng(42).integers(0, 255, (120, 160, 3), dtype=np.uint8)
        for index, row in enumerate(data.to_dict('records')):
            video = root / row['video']
            video.parent.mkdir(parents=True, exist_ok=True)
            video.write_bytes(str(index).encode())
            data.loc[index, 'video_sha256'] = p.b.sha(video)
            row['video_sha256'] = data.loc[index, 'video_sha256']
            cid = row['sample_id'] + '_f00000001'
            frame = source / 'candidate_frames' / f'{cid}.jpg'
            cv2.imwrite(str(frame), image)
            c = dict(candidate_id=cid, frame_index=1, frame_count=10,
                frame_path=frame.relative_to(source).as_posix(), frame_sha256=p.b.sha(frame),
                width=160, height=120, num_hands=1, box=[.1, .1, .8, .8], focus=100.,
                quality_status='eligible', quality_flags=[],
                landmarks=[[[float(x), float(y)] for x, y in zip(np.linspace(.2, .6, 21), np.linspace(.2, .7, 21))]])
            p.b.save_json(source / 'candidate_records' / f"{row['sample_id']}.json",
                dict(sample_id=row['sample_id'], video=row['video'], video_sha256=row['video_sha256'],
                     recipe='test', candidates=[c]))
            candidates.append(dict(sample_id=row['sample_id'], **{k: json.dumps(v) if isinstance(v, list) else v for k, v in c.items()}))
        data.to_csv(source / 'split_lock.csv', index=False)
        pd.DataFrame(candidates).to_csv(source / 'candidates.csv', index=False)
        p.b.save_json(source / 'candidate_config.json', dict(split_lock_sha256=p.b.sha(source / 'split_lock.csv'),
            quality_heuristics=p.f.QUALITY, annotation_recipe_sha256='test'))
        return source, data

    def test_end_to_end_prepare_blinding_import_analysis_and_tamper(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, data = self.fixture(root)
            first = p.sample_videos(data).iloc[0]
            (source / 'candidate_records' / f'{first.sample_id}.json').unlink()
            before = {str(path): p.b.sha(path) for path in source.rglob('*') if path.is_file()}
            work = root / 'pilot'
            manifest, selection = p.prepare(root, source, work)
            self.assertEqual(len(manifest), 160)
            self.assertEqual(selection.selection_status.eq('missing').sum(), 1)
            protocol = p.validate_frozen(work)
            self.assertEqual(protocol['known_signer_samples'], 160)
            for reviewer in p.REVIEWERS:
                html = (work / 'review' / f'{reviewer}.html').read_text(encoding='utf-8')
                for secret in ['train/', 'class_name', first.sample_id, 'video_sha256', 'candidate_frames/', 'candidate_id']:
                    self.assertNotIn(secret, html)
                self.assertIn('data:image/png;base64,', html)
                public = json.loads(html.split('const DATA=')[1].split(';\nconst categories=')[0])
                expected = sorted(public['samples'], key=lambda r: r['sample_id'])
                p.random.Random(43 if reviewer == 'reviewer_1' else 44).shuffle(expected)
                self.assertEqual([r['sample_id'] for r in public['samples']], [r['sample_id'] for r in expected])
            self.assertEqual(before, {str(path): p.b.sha(path) for path in source.rglob('*') if path.is_file()})
            with self.assertRaises(ValueError):
                p.prepare(root, source, work)
            missing = set(selection.loc[selection.selection_status.eq('missing'), 'sample_id'])
            for reviewer in p.REVIEWERS:
                payload = dict(schema=1, study_id=protocol['study_id'], reviewer_id=reviewer,
                    responses=[dict(sample_id=r.sample_id, decision='invalid' if r.sample_id in missing else r.class_name,
                        reasons=[], timestamp='2026-11-01T01:00:00Z') for r in manifest.itertuples()])
                response_file = root / f'{reviewer}.json'
                p.b.save_json(response_file, payload)
                if reviewer == 'reviewer_1':
                    bad = copy.deepcopy(payload)
                    next(r for r in bad['responses'] if r['sample_id'] in missing)['decision'] = 'saya'
                    bad_file = root / 'bad.json'
                    p.b.save_json(bad_file, bad)
                    with self.assertRaises(ValueError):
                        p.import_review(work, bad_file, reviewer)
                p.import_review(work, response_file, reviewer)
                with self.assertRaises(ValueError):
                    p.import_review(work, response_file, reviewer)
            p.analyze(work)
            summary = p.read_csv(work / 'analysis/pilot_summary.csv')
            self.assertTrue(summary.correct_n.eq(159).all())
            self.assertTrue((work / 'analysis/confusion_matrix.png').exists())
            selected = selection.loc[selection.selected_path.ne('')].iloc[0]
            original = p.read_csv(source / 'candidates.csv').set_index('candidate_id').loc[selected.candidate_id]
            self.assertTrue(np.array_equal(cv2.imread(str(source / original.frame_path)), cv2.imread(str(work / selected.selected_path))))
            (work / selected.selected_path).write_bytes(b'tampered')
            with self.assertRaises(ValueError):
                p.validate_frozen(work)

    @unittest.skipUnless(shutil.which('node'), 'Node unavailable')
    def test_javascript_navigation_draft_export_and_independence(self):
        html = p.review_html('study', 'reviewer_1', [dict(sample_id='opaque1', image='data:image/png;base64,AA=='),
                                                   dict(sample_id='opaque2', image='')])
        script = html.split('<script>')[1].split('</script>')[0]
        harness = r'''
const vm=require('vm'), assert=require('assert');
const elements={}, storage=new Map(), downloads=[];
function element(tag){return {tag,value:'',textContent:'',hidden:false,disabled:false,children:[],options:[],
 append(...items){this.children.push(...items);if(tag==='select')this.options.push(...items)},
 removeAttribute(name){delete this[name]},click(){downloads.push(this.download)}}}
for(const id of ['status','counter','sample','frame','missing','decision','reasons','prev','next','draft','export','import'])elements[id]=element(id==='decision'?'select':id);
const boxes=()=>elements.reasons.children.flatMap(l=>l.children).filter(x=>x.type==='checkbox');
const context={document:{getElementById:id=>elements[id],createElement:element,createTextNode:x=>x,
 querySelectorAll:selector=>selector.endsWith(':checked')?boxes().filter(x=>x.checked):boxes()},
 localStorage:{getItem:k=>storage.get(k),setItem:(k,v)=>storage.set(k,v)},
 Blob:class{},URL:{createObjectURL:()=> 'blob:test',revokeObjectURL:()=>{}},setTimeout:()=>{}};
vm.createContext(context);
vm.runInContext(SCRIPT,context);
assert(elements.counter.textContent.includes('1 / 2'));
elements.export.onclick();assert.equal(downloads.length,0);
elements.decision.value='saya';elements.decision.onchange();
elements.next.onclick();assert.equal(elements.missing.hidden,false);
assert(elements.decision.options.find(o=>o.value==='saya').disabled);
elements.decision.value='invalid';elements.decision.onchange();
elements.export.onclick();assert.equal(downloads.pop(),'reviewer_1_final.json');
elements.prev.onclick();assert.equal(elements.decision.value,'saya');
assert(storage.has('bisindo-pilot:study:reviewer_1'));
assert(!storage.has('bisindo-pilot:study:reviewer_2'));
assert.throws(()=>vm.runInContext("valid({schema:1,study_id:'study',reviewer_id:'reviewer_2',responses:[]},false)",context));
console.log('JS syntax/navigation/draft/final export gates passed');
'''.replace('SCRIPT', json.dumps(script))
        result = subprocess.run(['node'], input=harness, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_changed_video_and_catalog_cache_mismatch_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, data = self.fixture(root)
            record = p.sample_videos(data).iloc[0].to_dict()
            catalog = p.read_csv(source / 'candidates.csv')
            catalog = catalog[catalog.sample_id.eq(record['sample_id'])]
            config = json.loads((source / 'candidate_config.json').read_text())
            bad = catalog.copy()
            bad['frame_index'] = 999
            with self.assertRaises(ValueError):
                p.inspect_candidates(root, source, record, bad, config)
            (root / record['video']).write_bytes(b'changed')
            with self.assertRaises(ValueError):
                p.inspect_candidates(root, source, record, catalog, config)

    def test_constant_ratings_zero_identifiable_and_alignment(self):
        manifest = pd.DataFrame([dict(sample_id='opaque1', class_name='saya'), dict(sample_id='opaque2', class_name='kamu')])
        selection = pd.DataFrame([dict(sample_id=sid, selection_status='flagged') for sid in manifest.sample_id])
        ratings = [dict(row, decision='ambiguous') for row in self.payload()['responses']]
        summary, _, agreement, _ = p.calculate_metrics(manifest, selection, [ratings, ratings[::-1]])
        self.assertIsNone(summary[0]['accuracy_identifiable'])
        self.assertIsNone(agreement['cohens_kappa'])
        self.assertEqual(agreement['agreement'], 1.)
        with self.assertRaises(ValueError):
            p.calculate_metrics(manifest, selection.iloc[:1], [ratings, ratings])


if __name__ == '__main__':
    unittest.main()
