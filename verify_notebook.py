"""Execute all default-off notebook code locally, capturing report outputs.

Widget UI is constructed on demand by the user, not by this headless check.
"""
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from pathlib import Path
import nbformat

p=Path('BISINDO_YOLO26_Training_Set_Expansion.ipynb')
nb=nbformat.read(p,4)
namespace={'__name__':'__notebook__'}
count=0
for cell in nb.cells:
    if cell.cell_type!='code':
        continue
    count+=1
    outputs=[]
    def display_report(obj):
        if hasattr(obj,'to_html'):
            outputs.append(nbformat.v4.new_output('display_data',data={'text/html':obj.to_html(),'text/plain':str(obj)}))
        else:
            outputs.append(nbformat.v4.new_output('display_data',data={'text/plain':str(obj)}))
    source=cell.source
    if source.endswith('review_widget()'):
        source=source[:-len('review_widget()')] + 'print("Review widget siap; jalankan sel ini di notebook untuk membuka UI.")'
    stream=StringIO()
    with redirect_stdout(stream), redirect_stderr(stream):
        if 'from IPython.display import' in source:
            exec(source,namespace)
            namespace['display']=display_report
        else:
            exec(source,namespace)
    if stream.getvalue():
        outputs.insert(0,nbformat.v4.new_output('stream',name='stdout',text=stream.getvalue()))
    cell.outputs=outputs
    cell.execution_count=count
    print(f'PASS cell {count}',flush=True)
    # Save progress without losing reports if a later cell fails.
    nbformat.write(nb,p)
nbformat.validate(nb)
assert namespace['RUN_TRAINING'] is False
assert namespace['RUN_EVALUATION'] is False
assert not (namespace['WORK']/'runs').exists()
print('Default notebook flow passed. No training/evaluation executed.')
