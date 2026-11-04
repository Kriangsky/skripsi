"""Candidate frames are proposals. Only explicit human gesture review creates positives."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json
import shutil
import threading

import cv2
import numpy as np
import pandas as pd

import bisindo_pipeline as b

PRIMARY = b.FRAME_PRIMARY
FALLBACK = b.FRAME_FALLBACK
QUALITY = b.FRAME_QUALITY
_thread = threading.local()

def indices_for(count, fractions):
    if count <= 0:
        return []
    return sorted({min(count-1,max(0,round((count-1)*fraction))) for fraction in fractions})

def read_frames(video, fractions):
    cap=cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            raise ValueError(f'Cannot open video: {video}')
        count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if count<=0:
            count=0
            while cap.read()[0]:count+=1
            cap.release();cap=cv2.VideoCapture(str(video))
        indices=indices_for(count,fractions)
        if not indices:raise ValueError('No readable frames')
        wanted=set(indices);frames={}
        for index in range(indices[-1]+1):
            ok,image=cap.read()
            if not ok:
                break
            if index in wanted:frames[index]=image
        return count,frames,sorted(wanted-set(frames))
    finally:
        cap.release()

def assess_hands(image, hands):
    """Technical flags only; no inference of gesture correctness or completeness."""
    h,w=image.shape[:2]
    flags=[];focus=[]
    if not hands:return None,0.,['no_hands']
    for hand in hands:
        points=np.array([[lm.x,lm.y] for lm in hand],dtype=float)
        if len(points)!=21 or not np.isfinite(points).all():
            flags.append('invalid_landmarks');continue
        lo,hi=points.min(axis=0),points.max(axis=0)
        if (lo<QUALITY['border_margin']).any() or (hi>1-QUALITY['border_margin']).any():
            flags.append('hand_near_border_or_clipped')
        width,height=(hi-lo)*[w,h]
        if min(width,height)<QUALITY['min_hand_side_px'] or width*height<QUALITY['min_hand_area_px']:
            flags.append('hand_too_small')
        x1,y1=np.maximum(0,np.floor(lo*[w,h])).astype(int)
        x2,y2=np.minimum([w,h],np.ceil(hi*[w,h])).astype(int)
        crop=image[y1:y2,x1:x2]
        if not crop.size:
            flags.append('empty_hand_roi');continue
        # Fixed ROI resolution makes this a comparable, but uncalibrated, blur heuristic.
        gray=cv2.cvtColor(cv2.resize(crop,(128,128)),cv2.COLOR_BGR2GRAY)
        sharpness=float(cv2.Laplacian(gray,cv2.CV_64F).var())
        focus.append(sharpness)
        if sharpness<QUALITY['min_focus']:flags.append('possible_hand_blur')
    try:
        box=b.union_box(hands)
        if box is None:flags.append('invalid_box')
        else:b.yolo_box(box)
    except ValueError:
        box=None;flags.append('invalid_box')
    return box,min(focus) if focus else 0.,sorted(set(flags))

def freeze_inventory(root,work):
    root,work=Path(root).resolve(),Path(work).resolve()
    work.mkdir(parents=True,exist_ok=True)
    lock=work/'split_lock.csv'
    if not lock.exists():
        established=root/'experiment_yolo26_v2/inventory.csv'
        if not established.exists():
            raise ValueError('Established inventory missing; supply split_lock.csv. Refusing to silently create new splits.')
        shutil.copy2(established,lock)
    return b.inventory(root,work)

def _contact_sheet(work,record,candidates):
    tiles=[]
    for c in candidates:
        path=work/c['frame_path'] if c.get('frame_path') else None
        image=cv2.imread(str(path)) if path and path.exists() else None
        tile=np.full((300,360,3),240,dtype=np.uint8)
        if image is not None:
            box=c.get('box')
            if box:
                h,w=image.shape[:2]
                cv2.rectangle(image,(round(box[0]*w),round(box[1]*h)),(min(w-1,round(box[2]*w)),min(h-1,round(box[3]*h))),(0,220,0),3)
            scale=min(360/image.shape[1],235/image.shape[0])
            image=cv2.resize(image,(max(1,round(image.shape[1]*scale)),max(1,round(image.shape[0]*scale))))
            tile[:image.shape[0],:image.shape[1]]=image
        color=(20,100,20) if c['quality_status']=='eligible' else (0,40,210)
        cv2.putText(tile,f"frame {c['frame_index']} | hands {c['num_hands']} | focus {c['focus']:.1f}",(5,252),cv2.FONT_HERSHEY_SIMPLEX,.42,color,1)
        reason=', '.join(c['quality_flags']) or 'technical checks only; verify gesture'
        cv2.putText(tile,reason[:50],(5,275),cv2.FONT_HERSHEY_SIMPLEX,.37,color,1)
        tiles.append(tile)
    if not tiles:
        tiles=[np.full((300,360,3),180,dtype=np.uint8)]
    while len(tiles)%3:tiles.append(np.full_like(tiles[0],240))
    canvas=np.vstack([np.hstack(tiles[i:i+3]) for i in range(0,len(tiles),3)])
    header=np.full((45,canvas.shape[1],3),255,dtype=np.uint8)
    cv2.putText(header,f"{record['class_name']} | {record['video']} | candidates, NOT approved positives",(8,28),cv2.FONT_HERSHEY_SIMPLEX,.5,(20,20,20),1)
    target=work/'contact_sheets'/f"{record['sample_id']}.jpg"
    if not cv2.imwrite(str(target),np.vstack([header,canvas])):raise RuntimeError('Cannot save contact sheet')

def _process_video(root,work,record,extended=False,refresh=False):
    import mediapipe as mp
    path=work/'candidate_records'/f"{record['sample_id']}.json"
    previous=json.loads(path.read_text()) if path.exists() else None
    fractions=sorted(set(PRIMARY+FALLBACK)) if extended else PRIMARY
    if previous and not refresh and previous['video_sha256']==record['video_sha256'] and previous['recipe']==b.annotation_recipe() and (not extended or previous.get('extended')):
        if all((not c.get('frame_path')) or ((work/c['frame_path']).exists() and b.sha(work/c['frame_path'])==c['frame_sha256']) for c in previous['candidates']):
            if not (work/'contact_sheets'/f"{record['sample_id']}.jpg").exists():_contact_sheet(work,record,previous['candidates'])
            return previous
    def process(fractions):
        count,frames,failed=read_frames(b.safe_path(root,record['video']),fractions)
        result=[]
        for index in indices_for(count,fractions):
            candidate_id=f"{record['sample_id']}_f{index:08d}"
            row=dict(candidate_id=candidate_id,frame_index=index,frame_count=count,frame_path='',frame_sha256='',
                width=0,height=0,num_hands=0,box=None,focus=0.,quality_status='flagged',quality_flags=[],landmarks=[])
            try:
                if index in failed:raise ValueError('frame_decode_failed')
                frame=work/'candidate_frames'/f'{candidate_id}.jpg'
                if not cv2.imwrite(str(frame),frames[index]):raise ValueError('frame_write_failed')
                image=cv2.imread(str(frame))
                row.update(frame_path=frame.relative_to(work).as_posix(),frame_sha256=b.sha(frame),width=image.shape[1],height=image.shape[0])
                output=_thread.detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)))
                box,focus,flags=assess_hands(image,output.hand_landmarks)
                row.update(num_hands=len(output.hand_landmarks),box=box,focus=focus,quality_flags=flags,
                    quality_status='eligible' if not flags else 'flagged',
                    landmarks=[[[float(lm.x),float(lm.y)] for lm in hand] for hand in output.hand_landmarks if all(np.isfinite([lm.x,lm.y]).all() for lm in hand)])
            except Exception as exc:
                row['quality_flags']=[str(exc)]
            result.append(row)
        return result
    error=''
    try:
        candidates=process(fractions)
        # Retry other timestamps if every middle candidate fails technical checks.
        if not extended and not any(c['quality_status']=='eligible' for c in candidates):
            extra=process(FALLBACK)
            candidates=sorted({c['frame_index']:c for c in candidates+extra}.values(),key=lambda c:c['frame_index'])
            extended=True
    except Exception as exc:
        candidates=[];error=str(exc)
    result=dict(sample_id=record['sample_id'],video=record['video'],video_sha256=record['video_sha256'],
        recipe=b.annotation_recipe(),extended=extended,candidates=candidates,error=error)
    temporary=path.with_suffix('.tmp');b.save_json(temporary,result);temporary.replace(path)
    _contact_sheet(work,record,candidates)
    return result

def prepare(root,work,more_samples=None,workers=4,refresh=False):
    """Generate candidate sets; a recommendation never becomes a positive automatically."""
    import mediapipe as mp
    cv2.setNumThreads(1)
    root,work=Path(root).resolve(),Path(work).resolve()
    df=freeze_inventory(root,work)
    for folder in ['candidate_frames','candidate_records','contact_sheets','frames','candidate_labels','overlays']:
        (work/folder).mkdir(exist_ok=True)
    config=dict(protocol='human_selected_positive_v1',primary_fractions=PRIMARY,fallback_fractions=FALLBACK,
        quality_heuristics=QUALITY,annotation_recipe_sha256=b.annotation_recipe(),split_lock_sha256=b.sha(work/'split_lock.csv'),
        note='Technical visibility/blur checks are heuristics; only human class-aware review establishes a gesture positive.')
    b.save_json(work/'candidate_config.json',config)
    asset=b.get_hand_asset(root,work)
    requested=set(more_samples or [])
    if requested-set(df.sample_id):raise ValueError('Unknown sample ID requested for more frames')
    records=df.to_dict('records')
    if requested:records=[r for r in records if r['sample_id'] in requested]
    detectors=[];guard=threading.Lock()
    def initialize():
        options=mp.tasks.vision.HandLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(asset)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,num_hands=2,min_hand_detection_confidence=.5,min_hand_presence_confidence=.5)
        _thread.detector=mp.tasks.vision.HandLandmarker.create_from_options(options)
        with guard:detectors.append(_thread.detector)
    try:
        with ThreadPoolExecutor(max_workers=workers,initializer=initialize) as pool:
            pending=[pool.submit(_process_video,root,work,r,r['sample_id'] in requested,refresh) for r in records]
            for number,future in enumerate(as_completed(pending),1):
                future.result()
                if number%50==0 or number==len(records):print(f'Candidate videos: {number}/{len(records)}',flush=True)
    finally:
        for detector in detectors:detector.close()
    catalogs=[]
    old_path=work/'annotations.csv'
    old=pd.read_csv(old_path,keep_default_na=False).set_index('sample_id').to_dict('index') if old_path.exists() else {}
    manifest=[]
    for record in df.to_dict('records'):
        cached=work/'candidate_records'/f"{record['sample_id']}.json"
        result=json.loads(cached.read_text()) if cached.exists() else dict(candidates=[],error='not_processed')
        candidates=result['candidates']
        for c in candidates:catalogs.append(dict(sample_id=record['sample_id'],**c))
        eligible=[c for c in candidates if c['quality_status']=='eligible']
        recommended=max(eligible,key=lambda c:c['focus'])['candidate_id'] if eligible else ''
        prior=old.get(record['sample_id'])
        chosen=next((c for c in candidates if prior and c['candidate_id']==prior.get('selected_candidate_id')),None)
        preserve=prior and chosen and chosen['frame_sha256']==prior.get('frame_sha256') and prior.get('annotation_recipe_sha256')==b.annotation_recipe() and prior.get('video_sha256')==record['video_sha256']
        if preserve:
            record.update({k:v for k,v in prior.items() if k not in record})
        else:
            record.update(frame_index=-1,selected_candidate_id='',frame_sha256='',width=0,height=0,num_hands=0,
                x1='',y1='',x2='',y2='',auto_status='needs_review',annotation_error=result.get('error',''),
                review_status='needs_review',frame_review_status='needs_review',reviewer='',review_note='',
                review_frame_sha256='',review_box_sha256='',review_selection_sha256='',quality_override_reason='',
                rejection_reason='no_technically_eligible_candidate' if not eligible else 'gesture_not_yet_verified',
                rejected_candidates=prior.get('rejected_candidates','{}') if prior else '{}',
                annotation_recipe_sha256=b.annotation_recipe())
            if prior and prior.get('video_sha256')==record['video_sha256'] and prior.get('annotation_recipe_sha256')==b.annotation_recipe():
                for field in ['rejection_reason','review_note','reviewer','rejected_candidates']:
                    record[field]=prior.get(field,record[field])
        record['recommended_candidate_id']=recommended
        record['candidate_count']=len(candidates)
        record['candidate_search_extended']=bool(result.get('extended',False))
        manifest.append(record)
    pd.DataFrame(catalogs).assign(
        box=lambda d:d.box.map(json.dumps),quality_flags=lambda d:d.quality_flags.map(json.dumps),
        landmarks=lambda d:d.landmarks.map(json.dumps)).to_csv(work/'candidates.csv',index=False) if catalogs else pd.DataFrame(columns=['sample_id','candidate_id']).to_csv(work/'candidates.csv',index=False)
    result=pd.DataFrame(manifest)
    for r in result[result.frame_review_status.ne('approved')].itertuples():
        for folder,ext in [('frames','jpg'),('candidate_labels','txt'),('overlays','jpg')]:
            stale=work/folder/f'{r.sample_id}.{ext}'
            if stale.exists():stale.unlink()
    temporary=work/'annotations.prepare.tmp';result.to_csv(temporary,index=False);temporary.replace(old_path)
    if not (work/'checkpoint.json').exists() and (root/'experiment_yolo26_v2/checkpoint.json').exists():
        shutil.copy2(root/'experiment_yolo26_v2/checkpoint.json',work/'checkpoint.json')
    render_report(work)
    return result

def catalog(work):
    df=pd.read_csv(Path(work)/'candidates.csv',keep_default_na=False)
    result={}
    for row in df.to_dict('records'):
        for field in ['box','quality_flags','landmarks']:row[field]=json.loads(row[field])
        result[row['candidate_id']]=row
    return result

def selection_digest(row):
    import hashlib
    return hashlib.sha256(json.dumps(dict(candidate=row['selected_candidate_id'],frame_index=int(row['frame_index']),
        class_id=int(row['class_id']),video_sha256=row['video_sha256'],frame_sha256=row['frame_sha256']),sort_keys=True).encode()).hexdigest()

def selection_issues(work,df):
    work=Path(work)
    candidates=catalog(work)
    errors=[]
    for row in df.to_dict('records'):
        key=row['sample_id'];chosen=candidates.get(row.get('selected_candidate_id'))
        if row.get('frame_review_status')!='approved' or not chosen:
            errors.append(f'{key}: no human-verified representative gesture frame');continue
        if chosen['sample_id']!=key or int(row['frame_index'])!=chosen['frame_index'] or row['frame_sha256']!=chosen['frame_sha256']:
            errors.append(f'{key}: selected candidate/frame mismatch')
        if row.get('review_selection_sha256')!=selection_digest(row):errors.append(f'{key}: frame selection changed since review')
        if chosen['quality_status']!='eligible' and not str(row.get('quality_override_reason','')).strip():
            errors.append(f'{key}: technical flags lack human override reason')
        if not chosen.get('frame_path') or not (work/chosen['frame_path']).exists() or b.sha(work/chosen['frame_path'])!=chosen['frame_sha256']:
            errors.append(f'{key}: candidate JPEG changed/missing')
    return errors

def import_review(root,work,review_json):
    """Apply explicit frame decisions; unresolved videos stay needs_review, never dropped."""
    root,work=Path(root).resolve(),Path(work).resolve()
    payload=json.loads(Path(review_json).read_text(encoding='utf-8'))
    if payload.get('schema')!=1 or payload.get('kind')!='frame_selection':raise ValueError('Expected frame_selection export')
    if payload.get('annotations_sha256')!=b.sha(work/'annotations.csv') or payload.get('candidates_sha256')!=b.sha(work/'candidates.csv'):
        raise ValueError('Stale frame review; regenerate HTML from the current manifest')
    reviewer=payload.get('reviewer','')
    if not isinstance(reviewer,str) or not reviewer.strip():raise ValueError('Reviewer required')
    df=pd.read_csv(work/'annotations.csv',keep_default_na=False)
    candidates=catalog(work)
    decisions=payload.get('records',[])
    ids=[r.get('sample_id') for r in decisions]
    if len(ids)!=len(set(ids)) or set(ids)!=set(df.sample_id):raise ValueError('Every video must appear exactly once')
    decisions={r['sample_id']:r for r in decisions}
    copies=[];more=[]
    for index,row in df.iterrows():
        decision=decisions[row.sample_id]
        if decision.get('class_id')!=int(row.class_id):raise ValueError('Class mapping cannot change during review')
        if b.sha(b.safe_path(root,row.video))!=row.video_sha256:raise ValueError('Source video changed')
        rejected=decision.get('rejected_candidates',{})
        if not isinstance(rejected,dict):raise ValueError('Invalid rejection reasons')
        for key,reason in rejected.items():
            if key not in candidates or candidates[key]['sample_id']!=row.sample_id or not isinstance(reason,str) or not reason.strip():
                raise ValueError('Rejection must reference a candidate of this video and include a reason')
        choice=decision.get('candidate_id')
        df.loc[index,'rejected_candidates']=json.dumps(rejected)
        if not choice:
            df.loc[index,['frame_review_status','review_status','auto_status']]=['needs_review']*3
            df.loc[index,['selected_candidate_id','frame_sha256','review_frame_sha256','review_box_sha256','review_selection_sha256','quality_override_reason']]=''
            df.loc[index,['frame_index','width','height','num_hands']]=[-1,0,0,0]
            df.loc[index,['x1','y1','x2','y2']]=''
            df.loc[index,'reviewer']=reviewer.strip()
            df.loc[index,'review_note']=decision.get('note','')
            df.loc[index,'rejection_reason']=decision.get('reason') or 'gesture_not_yet_verified'
            if decision.get('request_more') is True:more.append(row.sample_id)
            continue
        selected=candidates.get(choice)
        if not selected or selected['sample_id']!=row.sample_id or choice in rejected:raise ValueError('Invalid/rejected frame selection')
        if decision.get('gesture_verified') is not True:raise ValueError('Explicit verification of gesture/class is required')
        override=decision.get('quality_override_reason','')
        if not isinstance(override,str) or not isinstance(decision.get('note',''),str) or not isinstance(decision.get('reason',''),str):
            raise ValueError('Review notes and reasons must be text')
        if selected['quality_status']!='eligible' and (not isinstance(override,str) or not override.strip()):raise ValueError('Flagged candidate requires human justification')
        box=decision.get('box')
        if not isinstance(box,list) or len(box)!=4 or any(isinstance(v,bool) or not isinstance(v,(int,float)) for v in box):raise ValueError('Selected frame requires a numeric reviewed box')
        b.yolo_box(box)
        candidate_path=b.safe_path(work,selected['frame_path'])
        if b.sha(candidate_path)!=selected['frame_sha256']:raise ValueError('Candidate JPEG changed')
        df.loc[index,['selected_candidate_id','frame_index','frame_sha256','width','height','num_hands']]=[
            choice,selected['frame_index'],selected['frame_sha256'],selected['width'],selected['height'],selected['num_hands']]
        df.loc[index,['x1','y1','x2','y2']]=box
        df.loc[index,['frame_review_status','review_status','reviewer','review_note','review_frame_sha256','auto_status','quality_override_reason','rejection_reason']]=[
            'approved','corrected' if selected['box'] is None or not np.allclose(box,selected['box'],rtol=0,atol=1e-12) else 'approved',reviewer.strip(),decision.get('note',''),selected['frame_sha256'],'detected' if selected['num_hands'] else 'manual_box',override,'']
        df.loc[index,'review_box_sha256']=b.review_digest(df.loc[index])
        df.loc[index,'review_selection_sha256']=selection_digest(df.loc[index])
        copies.append((candidate_path,work/'frames'/f'{row.sample_id}.jpg'))
    fingerprint=b.sha(work/'annotations.csv')
    backup=work/f'annotations_before_selection_{fingerprint[:16]}.csv'
    if not backup.exists():shutil.copy2(work/'annotations.csv',backup)
    for source,target in copies:shutil.copy2(source,target)
    # Remove stale positives for unresolved videos; originals and candidate JPEGs are retained.
    for r in df[df.frame_review_status.ne('approved')].itertuples():
        for folder,ext in [('frames','jpg'),('candidate_labels','txt'),('overlays','jpg')]:
            p=work/folder/f'{r.sample_id}.{ext}'
            if p.exists():p.unlink()
    temp=work/'annotations.selection.tmp';df.to_csv(temp,index=False);temp.replace(work/'annotations.csv')
    b.save_json(work/f'frame_review_{fingerprint[:16]}.json',payload)
    b.save_json(work/'qc_approval.json',dict(approved=False,reason='Frame review imported; run dataset validation'))
    if more:prepare(root,work,more_samples=more)
    else:render_report(work)
    return pd.read_csv(work/'annotations.csv',keep_default_na=False)

def render_report(work):
    work=Path(work).resolve()
    df=pd.read_csv(work/'annotations.csv',keep_default_na=False)
    candidates=catalog(work)
    records=[]
    grouped={key:[] for key in df.sample_id}
    for candidate in candidates.values():
        item={k:v for k,v in candidate.items() if k!='landmarks'}
        item['hand_bounds']=[[float(np.min(hand,axis=0)[0]),float(np.min(hand,axis=0)[1]),float(np.max(hand,axis=0)[0]),float(np.max(hand,axis=0)[1])] for hand in candidate['landmarks'] if hand]
        if candidate['sample_id'] in grouped:grouped[candidate['sample_id']].append(item)
    for row in df.to_dict('records'):
        options=grouped[row['sample_id']]
        records.append(dict(sample_id=row['sample_id'],video=row['video'],class_name=row['class_name'],class_id=int(row['class_id']),
            source=row['source'],split=row['split'],recommended=row['recommended_candidate_id'],selected=row['selected_candidate_id'],
            extended=str(row.get('candidate_search_extended',False)).lower()=='true',
            rejection_reason=row['rejection_reason'],rejected_candidates=json.loads(row['rejected_candidates']),
            quality_override_reason=row['quality_override_reason'],review_note=row['review_note'],
            selected_box=[float(row[k]) for k in ['x1','y1','x2','y2']] if row['frame_review_status']=='approved' else None,
            candidates=options))
        label=work/'candidate_labels'/f"{row['sample_id']}.txt"
        if row['frame_review_status']=='approved':
            values=b.yolo_box([float(row[k]) for k in ['x1','y1','x2','y2']])
            label.write_text(f"{row['class_id']} "+' '.join(f'{v:.8f}' for v in values)+'\n')
        elif label.exists():label.unlink()
    payload=json.dumps(dict(annotations_sha256=b.sha(work/'annotations.csv'),candidates_sha256=b.sha(work/'candidates.csv'),records=records),allow_nan=False).replace('<','\\u003c')
    template=(Path(__file__).parent/'frame_review_template.html').read_text(encoding='utf-8')
    (work/'frame_review.html').write_text(template.replace('__DATASET_JSON__',payload),encoding='utf-8')
    eligible=sum(c['quality_status']=='eligible' for c in candidates.values())
    summary=dict(videos=len(df),candidate_frames=len(candidates),technically_eligible_candidates=eligible,
        flagged_candidates=len(candidates)-eligible,selected_positives=int(df.frame_review_status.eq('approved').sum()),
        needs_review=int(df.frame_review_status.ne('approved').sum()),training_allowed=False,
        note='Eligible means technical checks passed, NOT that the gesture matches the class. No positives without human selection.')
    b.save_json(work/'annotation_summary.json',summary)
    df.groupby(['class_name','source','split','frame_review_status']).size().rename('samples').to_csv(work/'annotation_counts.csv')
    df[df.frame_review_status.ne('approved')].to_csv(work/'needs_review.csv',index=False)
    pd.DataFrame([dict(sample_id=c['sample_id'],candidate_id=c['candidate_id'],frame_index=c['frame_index'],reason=';'.join(c['quality_flags'])) for c in candidates.values() if c['quality_status']!='eligible']).to_csv(work/'flagged_candidates.csv',index=False)
    return summary

if __name__=='__main__':
    root=Path(__file__).resolve().parent
    prepare(root,root/'experiment_yolo26_v3')
    b.record_environment(root/'experiment_yolo26_v3')
