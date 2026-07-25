#!/usr/bin/env python3
"""SkeletonGraph banner v2 — big typographic wordmark (Hermes-style), our identity."""
import math, random
from PIL import Image, ImageDraw, ImageFont, ImageFilter

S=2
W,H=1600,470
FW,FH=W*S,H*S
FL="/usr/share/fonts/truetype/liberation/"
FD="/usr/share/fonts/truetype/dejavu/"
def F(kind,size):
    m={"bold":FL+"LiberationSans-Bold.ttf","reg":FL+"LiberationSans-Regular.ttf",
       "mono":FD+"DejaVuSansMono.ttf","monob":FD+"DejaVuSansMono-Bold.ttf"}
    return ImageFont.truetype(m[kind],int(size*S))
def x(v): return int(round(v*S))

TEAL=(56,205,168); TEAL_HI=(180,250,225); CYAN=(120,230,245)
DEEP=(34,168,168); EMER=(120,244,200)
AMBER=(247,183,74); AMBER_HI=(255,214,150)
WHITE=(238,243,249); MUT=(150,166,186)

def vgrad(size,top,bot):
    w,h=size
    g=Image.new("RGB",(1,h))
    for j in range(h):
        t=j/max(1,h-1)
        g.putpixel((0,j),tuple(int(top[i]+(bot[i]-top[i])*t) for i in range(3)))
    return g.resize((w,h))

def main():
    # background: near-black with subtle radial teal glow + vignette
    base=Image.new("RGB",(FW,FH),(8,10,16)).convert("RGBA")
    gl=Image.new("RGBA",(FW,FH),(0,0,0,0)); dg=ImageDraw.Draw(gl)
    dg.ellipse([x(W*0.5-620),x(H*0.5-360),x(W*0.5+620),x(H*0.5+360)],fill=(30,120,120,55))
    dg.ellipse([x(W*0.5-300),x(H*0.5-220),x(W*0.5+300),x(H*0.5+220)],fill=(40,150,150,40))
    gl=gl.filter(ImageFilter.GaussianBlur(x(80)))
    base.alpha_composite(gl)

    # faint code texture
    tx=Image.new("RGBA",(FW,FH),(0,0,0,0)); dt=ImageDraw.Draw(tx)
    snip=["tree_sitter.parse(src) → AST","bm25 ∪ dense ∪ graph → rrf(k=60)",
          "pagerank(node) · callers · callees","return skeleton.functions[fqn]",
          "sg_search(\"drops wrong index\")","_alter_field → alter_field → schema"]
    random.seed(5)
    for i in range(12):
        dt.text((x(random.randint(30,1150)),x(random.randint(16,H-24))),
                random.choice(snip),font=F("mono",13),fill=(120,150,180,14))
    base.alpha_composite(tx)

    # ---- big wordmark ----
    word="SKELETONGRAPH"
    target=1440.0
    fs=180
    while F("bold",fs).getlength(word)/S>target and fs>40: fs-=2
    f=F("bold",fs)
    twid=f.getlength(word)/S
    asc,desc=f.getmetrics()
    thei=(asc)/S
    wx=(W-twid)/2; wy=118

    # text mask (full canvas)
    mask=Image.new("L",(FW,FH),0); dm=ImageDraw.Draw(mask)
    dm.text((x(wx),x(wy)),word,font=f,fill=255)

    # glow behind wordmark
    gm=mask.filter(ImageFilter.GaussianBlur(x(9)))
    glow=Image.new("RGBA",(FW,FH),(0,0,0,0))
    glow.paste((TEAL[0],TEAL[1],TEAL[2],150),(0,0),gm)
    glow=glow.filter(ImageFilter.GaussianBlur(x(6)))
    base.alpha_composite(glow)

    # gradient fill via mask
    grad=vgrad((FW,FH),EMER,DEEP).convert("RGBA")
    grad.putalpha(mask)
    base.alpha_composite(grad)

    # crisp inner highlight (top sliver) — redraw text slightly with bright stroke
    hi=Image.new("RGBA",(FW,FH),(0,0,0,0)); dh=ImageDraw.Draw(hi)
    dh.text((x(wx),x(wy)),word,font=f,fill=(0,0,0,0),stroke_width=x(1.4),
            stroke_fill=(TEAL_HI[0],TEAL_HI[1],TEAL_HI[2],150))
    base.alpha_composite(hi)

    # circuit echo: an offset outline of the wordmark, low alpha
    echo=Image.new("RGBA",(FW,FH),(0,0,0,0)); de=ImageDraw.Draw(echo)
    de.text((x(wx+4),x(wy+5)),word,font=f,fill=(0,0,0,0),stroke_width=x(1.0),
            stroke_fill=(CYAN[0],CYAN[1],CYAN[2],55))
    base.alpha_composite(echo)

    # ---- graph nodes riding the wordmark ----
    dn=ImageDraw.Draw(base,"RGBA")
    # sample points along the top and baseline of the word for nodes
    random.seed(3)
    yb=wy+thei*0.02   # near top
    ybase=wy+thei*0.86
    pts=[]
    for frac in [0.06,0.20,0.33,0.5,0.63,0.77,0.9]:
        pts.append((wx+twid*frac, yb+random.uniform(-6,4)))
    for frac in [0.13,0.42,0.7,0.85]:
        pts.append((wx+twid*frac, ybase+random.uniform(-2,8)))
    # edges between some nearby points
    edges=[(0,1),(1,7),(7,2),(2,3),(3,8),(8,4),(4,5),(5,9),(9,6),(6,10)]
    for (i,j) in edges:
        if i<len(pts) and j<len(pts):
            dn.line([x(pts[i][0]),x(pts[i][1]),x(pts[j][0]),x(pts[j][1])],
                    fill=(TEAL_HI[0],TEAL_HI[1],TEAL_HI[2],70),width=x(1.0))
    amber_pt=pts[3]
    # node glows
    ng=Image.new("RGBA",(FW,FH),(0,0,0,0)); dng=ImageDraw.Draw(ng)
    for k,(px,py) in enumerate(pts):
        col=AMBER if k==3 else TEAL_HI
        r=8 if k==3 else 4.5
        dng.ellipse([x(px-r*2),x(py-r*2),x(px+r*2),x(py+r*2)],fill=(col[0],col[1],col[2],120))
    ng=ng.filter(ImageFilter.GaussianBlur(x(4)))
    base.alpha_composite(ng)
    for k,(px,py) in enumerate(pts):
        col=AMBER_HI if k==3 else TEAL_HI
        r=5.5 if k==3 else 3.2
        dn.ellipse([x(px-r),x(py-r),x(px+r),x(py+r)],fill=(col[0],col[1],col[2],255))
        dn.ellipse([x(px-r*0.4),x(py-r*0.4),x(px+r*0.4),x(py+r*0.4)],fill=(255,255,255,230))
    dn.ellipse([x(amber_pt[0]-13),x(amber_pt[1]-13),x(amber_pt[0]+13),x(amber_pt[1]+13)],
               outline=(AMBER[0],AMBER[1],AMBER[2],150),width=x(1.4))

    # ---- tagline + kicker ----
    d=ImageDraw.Draw(base,"RGBA")
    ty=wy+thei+34
    # centered tagline
    tag="the exact function, not a pile of files"
    fT=F("reg",27); tw2=fT.getlength(tag)/S
    d.text((x((W-tw2)/2),x(ty)),tag,font=fT,fill=(WHITE[0],WHITE[1],WHITE[2],240))
    sub="ZERO-LLM  ·  TREE-SITTER  ·  BM25 + DENSE + GRAPH  ·  SERVED OVER MCP"
    fS=F("bold",14); sw=fS.getlength(sub)/S
    d.text((x((W-sw)/2),x(ty+40)),sub,font=fS,fill=(TEAL[0],TEAL[1],TEAL[2],230))

    out=base.convert("RGB").resize((W,H),Image.LANCZOS)
    out.save("/sessions/blissful-loving-albattani/mnt/outputs/skeletongraph_banner2.png")
    print("saved",out.size,"fs",fs)

if __name__=="__main__": main()
