"""rigid_spec.py: build(name, prefix, primer, primer2, ...) writes assets/mogs/specs/<name>_rigid.json from the
rocketeer_rigid template (tiers, finish pass, cloth) and the colours in the character's rig.json look. Run from the
repo root:  python -c "import sys; sys.path.insert(0, 'tools/gri'); import rigid_spec; rigid_spec.build('maul', 'ml', rigid_spec.DARKP, rigid_spec.MIDP)"
Primers: DARKP under white/light topcoats, GREYP or MIDP otherwise (never red oxide)."""
import json, copy, os
GRI_CHARS = os.environ.get("GRI_CHARS", "C:/Repos/GRI_Dan/tools/out/characters")
def lin(c): return [round(((x+0.055)/1.055)**2.4 if x>0.04045 else x/12.92,4) for x in c]
T=json.load(open('assets/mogs/specs/rocketeer_rigid.json'))
W=T['stacks']['MOG_rr_orange']['weather']; GW=T['stacks']['MOG_rr_gunmetal']['weather']
DARKP=[0.05,0.06,0.08]; GREYP=[0.1,0.1,0.097]; MIDP=[0.06,0.065,0.07]
def look(n): return json.load(open(f'{GRI_CHARS}/{n}/export/rig.json'))['look']
def paint(c, rough, primer, w): return {"from":"MECH_paint_red","colors":{"topcoat":lin(c),"primer":primer},"layers":{"topcoat":{"rough":rough}},"weather":w}
def build(n, p, pp, p2p, extra_w=None, glass_glow=False, cloth=None, tier_extra=None):
    L=look(n); w=dict(W, **(extra_w or {})); gw=dict(GW, **({k:v for k,v in (extra_w or {}).items() if k in ('use','dirt','flat_impacts','detail')}))
    st={f"MOG_{p}_paint": paint(L['paint']['color'], L['paint']['rough'], pp, w),
        f"MOG_{p}_paint2": paint(L['paint2']['color'], L['paint2']['rough'], p2p, w),
        f"MOG_{p}_gunmetal": {"from":"MECH_gunmetal","colors":{"topcoat":lin(L['gunmetal']['color']),"primer":GREYP},"weather":gw},
        f"MOG_{p}_rubber": {"from":"MECH_rubber","color":lin(L['rubber']['color'])},
        f"MOG_{p}_glow": {"from":"MECH_eye","color":lin(L['glow']['color'])},
        f"MOG_{p}_steel": {"from":"MECH_steel","color":lin(L['steel']['color'])}}
    if max(L['gunmetal']['color']) <= 0.25:
        # Near-black gunmetal (GRI-approved on Maul): bright primer and bare steel read as white streaks.
        g=st[f"MOG_{p}_gunmetal"]; g['colors']['primer']=[0.06,0.06,0.058]; g['weather']=dict(gw, ridge=0.6)
    st[f"MOG_{p}_glass"] = ({"from":"MECH_eye","color":lin(L['glass']['color'])} if glass_glow else
        {"from":"MECH_glass","color":lin(L['glass']['color']),"rough":0.05,"weather":{"dust":0.1,"dirt":0.1,"detail":0.15,"use":0.1}})
    st[f"MOG_{p}_cloth"]=cloth or dict(copy.deepcopy(T['stacks']['MOG_rr_cloth']), color=lin(L['cloth']['color']))
    s=copy.deepcopy(T); s['stacks']=st
    s['slots']={k:f"MOG_{p}_{k}" for k in ('paint','paint2','gunmetal','rubber','glow','glass','cloth','steel')}
    s['slots'].update(decal=f"MOG_{p}_paint", eye=f"MOG_{p}_glow"); s['targets']={f"MOG_{p}_paint":1.0}
    if tier_extra: s['tier_set']['minimal'].update(tier_extra)
    s.pop('livery_tune', None)
    if 'exposure' in s:
        s['exposure'] = dict(s['exposure'], dir=f'{GRI_CHARS}/{n}/export/exposure')
    json.dump(s, open(f'assets/mogs/specs/{n}_rigid.json','w'), indent=2)
