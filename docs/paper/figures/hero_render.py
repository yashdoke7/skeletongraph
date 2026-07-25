#!/usr/bin/env python3
"""SkeletonGraph hero animation -> frames (2x supersampled) for GIF/MP4.

Story (single left->right pipeline):
  1 INDEX  repo -> tree-sitter (no LLM) -> .skeletongraph (BM25 + jina-code vectors + call-graph/PageRank)
  2 QUERY  an issue drops in, fans into 3 legs (lexical / semantic / structural)
  3 FUSE   reciprocal-rank fusion -> the ONE right function
  4 SERVE  handed to the agent over MCP; outcome chips count up
"""
import os, math
from PIL import Image, ImageDraw, ImageFont

S = 2                      # supersample
W, H = 1280, 720
FW, FH = W*S, H*S
OUT = "/sessions/blissful-loving-albattani/mnt/outputs/frames2"
os.makedirs(OUT, exist_ok=True)

FONTDIR_L = "/usr/share/fonts/truetype/liberation/"
FONTDIR_D = "/usr/share/fonts/truetype/dejavu/"
_fc = {}
def font(kind, size):
    size = int(size*S)
    key = (kind, size)
    if key in _fc: return _fc[key]
    m = {
        "reg":  FONTDIR_L+"LiberationSans-Regular.ttf",
        "bold": FONTDIR_L+"LiberationSans-Bold.ttf",
        "mono": FONTDIR_D+"DejaVuSansMono.ttf",
        "monob":FONTDIR_D+"DejaVuSansMono-Bold.ttf",
    }
    f = ImageFont.truetype(m[kind], size)
    _fc[key] = f
    return f

# palette
BG_TOP=(247,249,252); BG_BOT=(237,242,249)
INK=(15,23,42); MUT=(100,116,139); FAINT=(148,163,184)
GREEN=(22,163,74); GREEN_L=(220,244,228)
BLUE=(37,99,235); BLUE_L=(224,233,252)
PURP=(124,58,237); PURP_L=(235,228,251)
ORANGE=(234,88,12); ORANGE_L=(252,235,222)
AMBER=(245,158,11)
CARD=(255,255,255); LINE=(214,222,232)

def lerp(a,b,t): return a+(b-a)*t
def mix(c1,c2,t): return tuple(int(round(lerp(c1[i],c2[i],t))) for i in range(3))
def clamp(x,a=0.0,b=1.0): return max(a,min(b,x))
def smooth(t): t=clamp(t); return t*t*(3-2*t)
def ease_out(t): t=clamp(t); return 1-(1-t)**3
def seg(t,a,b):  # local 0..1 progress within [a,b]
    if b<=a: return 1.0 if t>=b else 0.0
    return clamp((t-a)/(b-a))

def x(v): return int(round(v*S))
def bg():
    img = Image.new("RGB",(FW,FH),BG_TOP)
    top=Image.new("RGB",(1,FH))
    for j in range(FH):
        top.putpixel((0,j), mix(BG_TOP,BG_BOT, j/FH))
    img.paste(top.resize((FW,FH)),(0,0))
    return img

def rrect(d, box, r, fill=None, outline=None, width=1, alpha=255):
    x0,y0,x1,y1=[x(v) for v in box]; r=x(r)
    if fill is not None:
        f = fill if len(fill)==4 else (fill[0],fill[1],fill[2],alpha)
        d.rounded_rectangle([x0,y0,x1,y1],radius=r,fill=f)
    if outline is not None:
        d.rounded_rectangle([x0,y0,x1,y1],radius=r,outline=outline,width=max(1,x(width)))

def shadow(base, box, r, blur=12, alpha=34, dy=6):
    # real soft drop shadow via a small cropped Gaussian blur (cheap: card-sized tile only)
    from PIL import ImageFilter
    x0,y0,x1,y1=box
    pad=int(blur*1.8)
    tw_=x(int((x1-x0)+2*pad)); th_=x(int((y1-y0)+2*pad))
    tile=Image.new("RGBA",(tw_,th_),(0,0,0,0))
    td=ImageDraw.Draw(tile)
    td.rounded_rectangle([x(pad),x(pad),x(pad+(x1-x0)),x(pad+(y1-y0))],
                         radius=x(r),fill=(15,23,42,alpha))
    tile=tile.filter(ImageFilter.GaussianBlur(x(blur*0.6)))
    base.alpha_composite(tile,(x(x0-pad),x(y0-pad+dy)))

def text(d, pos, s, kind, size, fill, anchor="la", alpha=255):
    f=font(kind,size)
    col = fill if len(fill)==4 else (fill[0],fill[1],fill[2],alpha)
    d.text((x(pos[0]),x(pos[1])), s, font=f, fill=col, anchor=anchor)

def tw(s,kind,size):
    f=font(kind,size); return f.getlength(s)/S

def line(d,p0,p1,fill,width=2,alpha=255):
    col=fill if len(fill)==4 else (fill[0],fill[1],fill[2],alpha)
    d.line([x(p0[0]),x(p0[1]),x(p1[0]),x(p1[1])],fill=col,width=max(1,x(width)))

def dot(d,p,r,fill,alpha=255):
    col=fill if len(fill)==4 else (fill[0],fill[1],fill[2],alpha)
    d.ellipse([x(p[0]-r),x(p[1]-r),x(p[0]+r),x(p[1]+r)],fill=col)

def star(d, cx, cy, r, color, alpha=255):
    pts=[]
    for i in range(10):
        ang=-math.pi/2 + i*math.pi/5
        rad = r if i%2==0 else r*0.45
        pts.append((x(cx+rad*math.cos(ang)), x(cy+rad*math.sin(ang))))
    d.polygon(pts, fill=(color[0],color[1],color[2],alpha))

def glow(base, p, r, color, alpha=90):
    # soft halo via a small cropped Gaussian blur (cheap: tiny tile only)
    from PIL import ImageFilter
    pad=int(r*0.9)
    size=x(2*(r+pad))
    tile=Image.new("RGBA",(size,size),(0,0,0,0))
    td=ImageDraw.Draw(tile)
    c=size//2
    td.ellipse([c-x(r),c-x(r),c+x(r),c+x(r)],fill=(color[0],color[1],color[2],alpha))
    tile=tile.filter(ImageFilter.GaussianBlur(x(r*0.55)))
    base.alpha_composite(tile,(x(p[0])-c,x(p[1])-c))

# ---- layout constants ----
def draw_folder(d, cx, cy, s, col, alpha=255):
    w=22*s; h=16*s
    tab_w=10*s; tab_h=4*s
    x0=cx-w/2; y0=cy-h/2
    d.rounded_rectangle([x(x0),x(y0-tab_h),x(x0+tab_w),x(y0+3*S/ S)],radius=x(2),fill=(col[0],col[1],col[2],alpha))
    d.rounded_rectangle([x(x0),x(y0),x(x0+w),x(y0+h)],radius=x(3),fill=(col[0],col[1],col[2],alpha))

def render(t, fade_in):
    base = bg().convert("RGBA")
    d = ImageDraw.Draw(base,"RGBA")

    # ---- header ----
    text(d,(64,44),"SkeletonGraph","bold",30,INK,anchor="lm",alpha=fade_in)
    # dot separator
    dot(d,(64+tw("SkeletonGraph","bold",30)+18,44),3.5,GREEN,alpha=fade_in)
    text(d,(64+tw("SkeletonGraph","bold",30)+34,45),
         "the exact function, not a pile of files","reg",16.5,MUT,anchor="lm",alpha=int(fade_in*0.95))
    text(d,(W-64,44),"zero-LLM structural retrieval  ·  MCP","mono",13.5,FAINT,anchor="rm",alpha=int(fade_in*0.9))
    line(d,(64,70),(W-64,70),LINE,1.4,alpha=int(fade_in*0.8))

    # stage kicker labels (top of pipeline)
    def kicker(px,label,num,col,a):
        text(d,(px,96),num,"bold",15,col,anchor="lm",alpha=a)
        text(d,(px+18,97),label,"bold",13,MUT,anchor="lm",alpha=a)

    # ---------- geometry ----------
    yC=330
    repo=(70,255,190,405)
    ts=(250,262,415,398)
    idx=(478,232,760,428)
    fuse_c=(905,330)             # RRF node center
    win=(985,258,1225,402)       # winner card

    # timeline phases
    p_index = seg(t,0.02,0.34)
    p_prompt= seg(t,0.34,0.50)
    p_legs  = seg(t,0.50,0.72)
    p_fuse  = seg(t,0.66,0.80)
    p_win   = seg(t,0.78,0.92)
    p_stats = seg(t,0.86,1.0)

    a_all=fade_in
    kicker(72,"INDEX  ·  once per repo, no LLM","1",GREEN,int(a_all*smooth(seg(t,0.0,0.08))))
    kicker(600,"QUERY  ·  one issue in","2",BLUE,int(a_all*smooth(seg(t,0.34,0.42))))
    kicker(980,"SERVE  ·  right function out","3",ORANGE,int(a_all*smooth(seg(t,0.72,0.80))))

    # ---------- 1. REPO ----------
    ra=int(a_all*smooth(seg(t,0.02,0.10)))
    if ra>4:
        shadow(base,repo,14)
        rrect(d,repo,14,fill=CARD); rrect(d,repo,14,outline=LINE,width=1.4)
        text(d,((repo[0]+repo[2])/2,repo[1]+24),"your repo","bold",15.5,INK,anchor="mm",alpha=ra)
        # file rows
        langs=["views.py","models.py","auth.go","api.ts"]
        for i,fn in enumerate(langs):
            fy=repo[1]+46+i*21
            fa=int(ra*smooth(seg(t,0.04+i*0.012,0.12+i*0.012)))
            rrect(d,(repo[0]+16,fy,repo[2]-16,fy+16),4,fill=(mix(CARD,BLUE_L,0.5)),alpha=fa)
            dot(d,(repo[0]+26,fy+8),2.6,BLUE,alpha=fa)
            text(d,(repo[0]+36,fy+8),fn,"mono",10.5,MUT,anchor="lm",alpha=fa)
        text(d,((repo[0]+repo[2])/2,repo[3]-14),"10 languages","reg",10.5,FAINT,anchor="mm",alpha=ra)

    # arrow repo->ts (files flowing)
    if p_index>0.02:
        line(d,(repo[2]+4,yC),(ts[0]-4,yC),LINE,2,alpha=ra)
        # flowing file dots
        for k in range(3):
            fp=((t*3.0+k/3.0)%1.0)
            if seg(t,0.06,0.34)>0:
                px=lerp(repo[2]+6,ts[0]-6,fp)
                dot(d,(px,yC),3.2,BLUE,alpha=int(ra*(1-abs(fp-0.5)*1.2)))

    # ---------- 2. TREE-SITTER ----------
    ta=int(a_all*smooth(seg(t,0.08,0.16)))
    if ta>4:
        shadow(base,ts,14)
        rrect(d,ts,14,fill=CARD); rrect(d,ts,14,outline=mix(LINE,GREEN,0.35),width=1.6)
        text(d,((ts[0]+ts[2])/2,ts[1]+24),"tree-sitter","bold",15.5,GREEN,anchor="mm",alpha=ta)
        text(d,((ts[0]+ts[2])/2,ts[1]+44),"parse → AST","reg",11.5,MUT,anchor="mm",alpha=ta)
        # a little call graph forming
        nodes=[(300,335),(340,320),(378,345),(330,368),(368,372)]
        na=int(ta*smooth(seg(t,0.14,0.30)))
        edges=[(0,1),(1,2),(0,3),(3,4),(2,4)]
        for (i,j) in edges:
            line(d,nodes[i],nodes[j],mix(LINE,GREEN,0.5),1.6,alpha=int(na*0.8))
        for i,n in enumerate(nodes):
            r = 5.5 if i in (0,2) else 4
            dot(d,n,r+0.5,CARD)
            dot(d,n,r,mix(GREEN,INK,0.0),alpha=na)
        text(d,((ts[0]+ts[2])/2,ts[2]-ts[2]+ts[1]+118),"functions + call graph","reg",10.5,FAINT,anchor="mm",alpha=na)

    # arrow ts->idx
    if p_index>0.35:
        line(d,(ts[2]+4,yC),(idx[0]-4,yC),LINE,2,alpha=ta)

    # ---------- 3. INDEX STORE ----------
    ia=int(a_all*smooth(seg(t,0.18,0.26)))
    if ia>4:
        shadow(base,idx,16)
        rrect(d,idx,16,fill=CARD); rrect(d,idx,16,outline=mix(LINE,GREEN,0.4),width=1.8)
        text(d,(idx[0]+20,idx[1]+24),".skeletongraph","monob",14.5,GREEN,anchor="lm",alpha=ia)
        text(d,(idx[2]-18,idx[1]+24),"rebuildable · no LLM","reg",10.5,FAINT,anchor="rm",alpha=ia)
        rows=[("BM25 index","lexical",BLUE,BLUE_L),
              ("jina-code vectors","semantic · hash-cached",PURP,PURP_L),
              ("call graph + PageRank","structural · centrality",GREEN,GREEN_L)]
        for i,(nm,sub,col,coll) in enumerate(rows):
            ry=idx[1]+48+i*46
            rowa=int(ia*smooth(seg(t,0.20+i*0.03,0.30+i*0.03)))
            rrect(d,(idx[0]+16,ry,idx[2]-16,ry+38),9,fill=coll,alpha=int(rowa*0.9))
            dot(d,(idx[0]+30,ry+19),4.5,col,alpha=rowa)
            text(d,(idx[0]+44,ry+13),nm,"bold",12.5,INK,anchor="lm",alpha=rowa)
            text(d,(idx[0]+44,ry+27),sub,"reg",10,MUT,anchor="lm",alpha=rowa)
            # build progress bar fill
            bx0,bx1=idx[2]-92,idx[2]-24
            fillp=smooth(seg(t,0.20+i*0.03,0.33+i*0.04))
            rrect(d,(bx0,ry+24,bx1,ry+29),2.5,fill=mix(CARD,col,0.25),alpha=rowa)
            if fillp>0:
                rrect(d,(bx0,ry+24,lerp(bx0,bx1,fillp),ry+29),2.5,fill=col,alpha=rowa)

    # ---------- 2. PROMPT drops in ----------
    pa=smooth(p_prompt)
    if pa>0.02:
        pb=(438,150,822,206)
        drop = lerp(-44,0,ease_out(seg(t,0.34,0.44)))
        pbb=(pb[0],pb[1]+drop,pb[2],pb[3]+drop)
        aa=int(a_all*pa)
        shadow(base,pbb,10,alpha=34)
        rrect(d,pbb,10,fill=mix(CARD,BLUE_L,0.25)); rrect(d,pbb,10,outline=mix(LINE,BLUE,0.4),width=1.5)
        text(d,(pbb[0]+16,pbb[1]+17),"issue","monob",10,BLUE,anchor="lm",alpha=aa)
        full='"_alter_field drops the wrong index on SQLite"'
        n=int(len(full)*ease_out(seg(t,0.37,0.50)))
        text(d,(pbb[0]+58,pbb[1]+17),full[:n],"mono",11.5,INK,anchor="lm",alpha=aa)
        text(d,(pbb[0]+16,pbb[1]+39),"natural language · symbols not guaranteed","reg",10,MUT,anchor="lm",alpha=int(aa*0.9))
        # connector prompt -> index
        if seg(t,0.47,0.55)>0:
            line(d,((pbb[0]+pbb[2])/2,pbb[3]),((pbb[0]+pbb[2])/2,idx[1]-4),mix(LINE,BLUE,0.4),1.6,alpha=int(aa*0.6))

    # ---------- 3. LEGS + FUSE ----------
    la=smooth(p_legs)
    legs=[("BM25",BLUE,idx[1]+67),("dense",PURP,idx[1]+113),("graph",GREEN,idx[1]+159)]
    if la>0.02:
        # three legs from index right edge converging to fuse node
        for i,(nm,col,ly) in enumerate(legs):
            p0=(idx[2]+4,ly)
            p1=(fuse_c[0]-46,fuse_c[1])
            aa=int(a_all*la)
            # curved-ish via midpoint
            line(d,p0,p1,mix(LINE,col,0.55),1.8,alpha=int(aa*0.85))
            # racing dots
            for k in range(2):
                fp=((t*3.2+k/2.0+i*0.15)%1.0)
                dd_a=int(aa*(1-abs(fp-0.5)*1.4))
                if dd_a>6:
                    px=lerp(p0[0],p1[0],fp); py=lerp(p0[1],p1[1],fp)
                    dot(d,(px,py),3.4,col,alpha=dd_a)
    # fuse node
    fa=smooth(p_fuse)
    if fa>0.02:
        aa=int(a_all*fa)
        fr=40
        glow(base,fuse_c,fr+10,AMBER,alpha=int(40*fa))
        d.ellipse([x(fuse_c[0]-fr),x(fuse_c[1]-fr),x(fuse_c[0]+fr),x(fuse_c[1]+fr)],
                  fill=(CARD[0],CARD[1],CARD[2],aa))
        d.ellipse([x(fuse_c[0]-fr),x(fuse_c[1]-fr),x(fuse_c[0]+fr),x(fuse_c[1]+fr)],
                  outline=(AMBER[0],AMBER[1],AMBER[2],aa),width=x(2.2))
        text(d,(fuse_c[0],fuse_c[1]-8),"RRF","bold",16,INK,anchor="mm",alpha=aa)
        text(d,(fuse_c[0],fuse_c[1]+12),"fuse · k=60","reg",10.5,MUT,anchor="mm",alpha=aa)
        line(d,(fuse_c[0]+fr+2,fuse_c[1]),(win[0]-4,fuse_c[1]),mix(LINE,AMBER,0.4),2,alpha=aa)

    # ---------- 4. WINNER ----------
    wa=smooth(p_win)
    if wa>0.02:
        pop=lerp(0.92,1.0,ease_out(seg(t,0.78,0.86)))
        cx=(win[0]+win[2])/2; cy=(win[1]+win[3])/2
        wb=(cx-(cx-win[0])*pop, cy-(cy-win[1])*pop, cx+(win[2]-cx)*pop, cy+(win[3]-cy)*pop)
        aa=int(a_all*wa)
        glow(base,(cx,cy),120,AMBER,alpha=int(30*wa))
        shadow(base,wb,16,alpha=48,dy=8)
        rrect(d,wb,16,fill=CARD); rrect(d,wb,16,outline=AMBER,width=2.2)
        star(d,wb[0]+24,wb[1]+22,7,AMBER,alpha=aa)
        text(d,(wb[0]+36,wb[1]+22),"rank 1","bold",12.5,AMBER,anchor="lm",alpha=aa)
        text(d,(wb[2]-16,wb[1]+22),"function-level","reg",10,MUT,anchor="rm",alpha=aa)
        text(d,(wb[0]+18,wb[1]+52),"db/backends/schema.py","mono",11.5,MUT,anchor="lm",alpha=aa)
        text(d,(wb[0]+18,wb[1]+74),"_alter_field()","monob",16,INK,anchor="lm",alpha=aa)
        rrect(d,(wb[0]+18,wb[1]+92,wb[2]-16,wb[1]+118),7,fill=GREEN_L,alpha=int(aa*0.9))
        text(d,(wb[0]+28,wb[1]+105),"→ served to the agent over MCP","reg",10.8,GREEN,anchor="lm",alpha=aa)

    # ---------- outcome chips ----------
    sa=smooth(p_stats)
    if sa>0.02:
        chips=[("first-search file recall","66% → 86%",GREEN),
               ("function pinpointed","0% → ~80%",BLUE),
               ("worst-case cost (p95)","−42%",ORANGE)]
        cw=356; gap=20; total=cw*3+gap*2; sx=(W-total)/2; cyb=470
        for i,(lab,val,col) in enumerate(chips):
            aa=int(a_all*smooth(seg(t,0.86+i*0.03,0.96+i*0.03)))
            if aa<4: continue
            cx0=sx+i*(cw+gap)
            box=(cx0,cyb,cx0+cw,cyb+60)
            shadow(base,box,12,alpha=26,dy=4)
            rrect(d,box,12,fill=CARD); rrect(d,box,12,outline=LINE,width=1.3)
            rrect(d,(cx0,cyb,cx0+5,cyb+60),3,fill=col,alpha=aa)
            text(d,(cx0+22,cyb+21),lab,"reg",12,MUT,anchor="lm",alpha=aa)
            text(d,(cx0+22,cyb+42),val,"bold",18,col if col!=ORANGE else ORANGE,anchor="lm",alpha=aa)

    # footer line
    text(d,(W/2,700),"index  ·  fuse lexical + semantic + structural  ·  return one function  ·  bound the agent’s wandering",
         "reg",12,FAINT,anchor="mm",alpha=int(a_all*0.9*smooth(seg(t,0.9,1.0))))

    if globals().get("POSTER"):
        return base.convert("RGB")
    out = base.convert("RGB").resize((W,H),Image.LANCZOS)
    return out

def main():
    import sys
    FPS=25; DUR=12.0
    N=int(FPS*DUR)
    a=int(sys.argv[1]) if len(sys.argv)>1 else 0
    b=int(sys.argv[2]) if len(sys.argv)>2 else N
    for i in range(a,b):
        t=i/N
        # global fade in first 6% and hold/loop; small fade at very end handled by loop
        fi = int(255*smooth(seg(t,0.0,0.05)))
        if t>0.97: fi=int(255*smooth(1-(t-0.97)/0.03))
        fi=max(0,min(255,fi if t>0.02 else int(255*smooth(t/0.05))))
        frame=render(t, max(20,fi) if 0.05<t<0.97 else fi)
        frame.save(f"{OUT}/f{i:04d}.png")
        if i%20==0: print("frame",i,"/",N)
    print("done",N,"frames")

if __name__=="__main__":
    main()
