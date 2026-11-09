"""python space_generate.py OUT_DIR IMAGE SEED N. A saved OUT_DIR/resp.json skips generation, so a stopped download can resume."""
import sys,json,os,time
import httpx
sys.stdout.reconfigure(encoding='utf8')
B='https://hyz317-hairdeck.hf.space/gradio_api'
out=sys.argv[1]; img=sys.argv[2]; seed=int(sys.argv[3]); n=int(sys.argv[4])
cl=httpx.Client(timeout=httpx.Timeout(600,connect=60),follow_redirects=True)

def retry(fn,tries=6):
    # the Space resets connections (WinError 10054, 2026-11-08)
    for i in range(tries):
        try: return fn()
        except httpx.TransportError as e:
            print('retry',i+1,repr(e)[:120],flush=True)
            if i==tries-1: raise
            time.sleep(5*(i+1))

def fetch(url,path,size=None):
    if not url.startswith(B.rsplit('/',1)[0]+'/'): raise ValueError('file URL is not on the Space: '+url)
    if size and os.path.exists(path) and os.path.getsize(path)==size: return
    def get():
        r=cl.get(url); r.raise_for_status()
        if size and len(r.content)!=size: raise httpx.ReadError(f'got {len(r.content)} of {size} bytes')
        return r.content
    data=retry(get)
    with open(path,'wb') as h: h.write(data)
    print(os.path.basename(path),len(data),flush=True)

resp=os.path.join(out,'resp.json')
if os.path.exists(resp):
    d=json.load(open(resp))
else:
    up=retry(lambda:cl.post(B+'/upload',files={'files':open(img,'rb')}).json()[0])
    fd={'path':up,'meta':{'_type':'gradio.FileData'},'orig_name':os.path.basename(img)}
    t=time.time()
    eid=retry(lambda:cl.post(B+'/call/generate',json={'data':[fd,seed,50,2.0,n,False,True]}).json()['event_id'])
    print('event',eid,flush=True)
    last=None
    with cl.stream('GET',B+'/call/generate/'+eid) as r:
        ev=None
        for line in r.iter_lines():
            if line.startswith('event:'): ev=line[6:].strip(); print('ev',ev,round(time.time()-t,1),flush=True) if ev!='heartbeat' else None
            elif line.startswith('data:') and ev in('complete','error'):
                last=(ev,line[5:].strip())
    print('total s',round(time.time()-t,1),last[0])
    if last[0]=='error': print(last[1][:800]); sys.exit(1)
    d=json.loads(last[1])
    json.dump(d,open(resp,'w'),indent=1)
print(repr(d[6])[:300])
# ponytail: only the JSON files, because rig.py reads them and no step reads the GLB files
for f in d[5] or []:
    name=os.path.basename(f.get('orig_name') or f['path'])
    if name.endswith('.json'): fetch(f['url'],os.path.join(out,name),f.get('size'))
if d[0]: fetch(d[0]['url'],os.path.join(out,'prep.webp'))
