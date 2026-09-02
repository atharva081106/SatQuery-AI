import { jsx as _jsx } from "react/jsx-runtime";
const useIsStaticRenderer = () => false;
import{useEffect,useRef}from"react";// ───────────────────────────────────────────────────────── core/math
/**
 * Shared math for all effects. Deliberately dependency-free.
 *//** Deterministic pseudo-random between 0 and 1. Same input, same output. */function hash01(a,b){const h=Math.sin(a*127.1+b*311.7)*43758.5453;return h-Math.floor(h);}/** Smooth 0→1 ramp. Avoids the hard edge a linear fade produces. */function smoothstep(edge0,edge1,x){const t=Math.min(1,Math.max(0,(x-edge0)/(edge1-edge0)));return t*t*(3-2*t);}function clamp(x,min,max){return Math.min(max,Math.max(min,x));}/**
 * Positive modulo. In JS `-10 % 3` is `-1`; we want `2`.
 * This is the heart of any infinitely wrapping grid.
 */function wrap(value,size){return(value%size+size)%size;}function identity(){return[1,0,0,0,1,0,0,0,1];}/** a × b */function multiply(a,b){return[a[0]*b[0]+a[1]*b[3]+a[2]*b[6],a[0]*b[1]+a[1]*b[4]+a[2]*b[7],a[0]*b[2]+a[1]*b[5]+a[2]*b[8],a[3]*b[0]+a[4]*b[3]+a[5]*b[6],a[3]*b[1]+a[4]*b[4]+a[5]*b[7],a[3]*b[2]+a[4]*b[5]+a[5]*b[8],a[6]*b[0]+a[7]*b[3]+a[8]*b[6],a[6]*b[1]+a[7]*b[4]+a[8]*b[7],a[6]*b[2]+a[7]*b[5]+a[8]*b[8]];}function rotationX(a){const c=Math.cos(a);const s=Math.sin(a);return[1,0,0,0,c,-s,0,s,c];}function rotationY(a){const c=Math.cos(a);const s=Math.sin(a);return[c,0,s,0,1,0,-s,0,c];}function rotationZ(a){const c=Math.cos(a);const s=Math.sin(a);return[c,-s,0,s,c,0,0,0,1];}/** Applies the matrix to a point. Writes into `out` to avoid allocating. */function transform(m,x,y,z,out){out.x=m[0]*x+m[1]*y+m[2]*z;out.y=m[3]*x+m[4]*y+m[5]*z;out.z=m[6]*x+m[7]*y+m[8]*z;}/**
 * Gram-Schmidt: makes the rows perpendicular and unit length again.
 *
 * A matrix multiplied by itself thousands of times slowly skews from rounding
 * error, and the sphere starts to visibly distort. Straightening it out
 * periodically keeps that away.
 */function orthonormalize(m){let[ax,ay,az,bx,by,bz]=m;let la=Math.hypot(ax,ay,az)||1;ax/=la;ay/=la;az/=la;// make b perpendicular to a
const dot=bx*ax+by*ay+bz*az;bx-=ax*dot;by-=ay*dot;bz-=az*dot;const lb=Math.hypot(bx,by,bz)||1;bx/=lb;by/=lb;bz/=lb;// c follows from the cross product, so it is correct by construction
const cx=ay*bz-az*by;const cy=az*bx-ax*bz;const cz=ax*by-ay*bx;return[ax,ay,az,bx,by,bz,cx,cy,cz];}/** Always produces a clean list, without empty entries. */function normalizeImages(input){if(!input)return[];const out=[];for(const item of input){if(!item)continue;if(typeof item==="string"){if(item)out.push({src:item});continue;}if(item.src)out.push({src:item.src,srcSet:item.srcSet,alt:item.alt});}return out;}/**
 * srcSet is not merely useless without `sizes`, it is harmful: the browser
 * assumes 100vw and picks the largest candidate. A 200px tile was pulling in
 * a 1400px image.
 *
 * The value is rounded to 50px steps so the attribute doesn't change every
 * frame while a tile grows or shrinks.
 */function updateSizes(img,cssPx){const bucket=Math.max(50,Math.ceil(cssPx/50)*50);const value=`${bucket}px`;if(img.getAttribute("sizes")!==value)img.sizes=value;}/** Sets source, srcset and alt at once, touching nothing that hasn't changed. */function applyImage(img,source,cssPx){// sizes before srcset: otherwise the browser starts loading on the wrong
// assumption and then fetches a second candidate
if(cssPx!==undefined)updateSizes(img,cssPx);if(img.getAttribute("src")!==source.src)img.src=source.src;const srcSet=source.srcSet||"";if(img.getAttribute("srcset")!==srcSet){if(srcSet)img.srcset=srcSet;else img.removeAttribute("srcset");}const alt=source.alt||"";if(img.alt!==alt)img.alt=alt;}/** Stable key for detecting whether the image list actually changed. */function imagesKey(images){return images.map(i=>i.src).join("|");}/**
 * Stand-in tiles for when no images are set yet.
 *
 * A component dropped on the Framer canvas renders nothing until the user
 * fills the image array, which reads as broken — and it is the first thing
 * anyone sees of the component. These are inline SVG data URIs, so they cost
 * no network request and no bundled assets, and they disappear the moment a
 * real image is added.
 *
 * Plain white cards, but not flat white: `tileShadow` is empty by default, so
 * without the faint gradient and hairline edge overlapping cards would merge
 * into one white mass.
 */function placeholderImages(count=12){const svg=`<svg xmlns="http://www.w3.org/2000/svg" width="400" height="400" viewBox="0 0 400 400">`+`<defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1">`+`<stop offset="0" stop-color="#ffffff"/><stop offset="1" stop-color="#ececed"/>`+`</linearGradient></defs>`+`<rect width="400" height="400" fill="url(#g)"/>`+`<rect x="1" y="1" width="398" height="398" fill="none" stroke="#000000" stroke-opacity=".07" stroke-width="2"/>`+`</svg>`;const src="data:image/svg+xml;utf8,"+encodeURIComponent(svg);// one identical card, repeated — no variation to distract from the motion
return Array.from({length:count},()=>({src,alt:""}));}// ───────────────────────────────────────────────────────── core/a11y
/**
 * Accessibility helpers every effect needs.
 *//**
 * A sphere that keeps spinning on its own is exactly what this system setting
 * exists for: sustained motion makes some people nauseous. Motion the user
 * initiates themselves — dragging — keeps working.
 */function prefersReducedMotion(){if(typeof window==="undefined"||!window.matchMedia)return false;return window.matchMedia("(prefers-reduced-motion: reduce)").matches;}/** Calls `cb` when the preference changes. Returns a cleanup function. */function onReducedMotionChange(cb){if(typeof window==="undefined"||!window.matchMedia)return()=>{};const mq=window.matchMedia("(prefers-reduced-motion: reduce)");const handler=()=>cb(mq.matches);mq.addEventListener("change",handler);return()=>mq.removeEventListener("change",handler);}/**
 * Visible focus ring on the tile that Enter would open.
 *
 * Cannot be an `outline`: tiles use `overflow:hidden` and sit under
 * transforms, so an outline gets clipped or hidden behind neighbours. An
 * inset box-shadow lives inside the element and stays visible.
 */const FOCUS_RING="inset 0 0 0 3px #fff, inset 0 0 0 5px rgba(0,0,0,.55)";/**
 * Offered, not the default. Out of the box tiles are flat — a shadow is a
 * styling choice, and one that should be the user's rather than ours.
 */const SOFT_TILE_SHADOW="0px 8px 30px 0px rgba(0,0,0,0.45)";const DEFAULT_OPEN_SHADOW="0px 30px 80px 0px rgba(0,0,0,0.6)";/**
 * Stacks the focus ring on top of whatever shadow the user configured.
 * An empty shadow is valid — then only the ring remains.
 */function composeShadow(base,focused){if(!focused)return base||"none";return base?`${base}, ${FOCUS_RING}`:FOCUS_RING;}// ────────────────────────────────────────────────────── core/pointer
/**
 * Tap detection for draggable effects.
 *
 * Without a threshold, any drag that happens to end on a tile opens it. And
 * note: `setPointerCapture` retargets every subsequent pointer event to the
 * container, so the tapped element must be captured on pointer*down* — by
 * pointerup, `e.target` no longer points at it.
 *//** Movement in px above which a gesture is a drag, not a tap. */const TAP_SLOP=6;/** Duration in ms above which a tap is no longer a tap. */const TAP_TIMEOUT=500;function createTapTracker(){let downX=0;let downY=0;let downT=0;let travelled=0;let hit=null;return{down(e,target){downX=e.clientX;downY=e.clientY;downT=performance.now();travelled=0;hit=target;},move(e){travelled=Math.hypot(e.clientX-downX,e.clientY-downY);},up(){const was=hit;hit=null;if(!was)return null;if(travelled>=TAP_SLOP)return null;if(performance.now()-downT>=TAP_TIMEOUT)return null;return was;}};}const lightboxDefaults={openScale:.8,openDuration:520,backdropOpacity:.72,backdropBlur:10,cornerRadius:6,openShadow:DEFAULT_OPEN_SHADOW,respectReducedMotion:true};/**
 * Well above anything an effect uses itself. Effects stack tiles by depth —
 * infinite-orbit up to 100, sphere-orbit up to 1000 — and a lightbox landing
 * in that range disappears behind the tiles.
 */const BACKDROP_Z=999998;const DETAIL_Z=999999;function createLightbox(root,userOptions={}){let o={...lightboxDefaults,...userOptions};let detail=null;let detailImg=null;let backdrop=null;let current=null;let returnFocusTo=null;let closeTimer=null;function ensureOverlay(){if(detail)return;backdrop=document.createElement("div");backdrop.style.cssText=`position:absolute;inset:0;z-index:${BACKDROP_Z};opacity:0;pointer-events:none;`+"cursor:zoom-out;transition:opacity var(--lb-dur) ease";backdrop.addEventListener("click",onBackdropClick);backdrop.setAttribute("aria-hidden","true");root.appendChild(backdrop);detail=document.createElement("div");detail.style.cssText=`position:absolute;z-index:${DETAIL_Z};overflow:hidden;opacity:0;pointer-events:none;`+"will-change:left,top,width,height";// a dialog that takes focus: screen readers announce it, and Escape
// works even when the click came from a keyboard user
detail.setAttribute("role","dialog");detail.setAttribute("aria-modal","true");detail.tabIndex=-1;detailImg=document.createElement("img");detailImg.style.cssText="width:100%;height:100%;object-fit:contain;display:block;pointer-events:none";detailImg.draggable=false;detailImg.alt="";detail.appendChild(detailImg);root.appendChild(detail);}/** An element's rect, relative to the container. */function relativeRect(el){const a=el.getBoundingClientRect();const b=root.getBoundingClientRect();return{left:a.left-b.left,top:a.top-b.top,w:a.width,h:a.height};}/** Centred, within openScale, preserving aspect ratio. */function targetRect(img){const W=root.clientWidth;const H=root.clientHeight;const ar=img.naturalWidth&&img.naturalHeight?img.naturalWidth/img.naturalHeight:1;const maxW=W*o.openScale;const maxH=H*o.openScale;let w=maxW;let h=w/ar;if(h>maxH){h=maxH;w=h*ar;}return{left:(W-w)/2,top:(H-h)/2,w,h};}function applyRect(el,r){el.style.left=`${r.left}px`;el.style.top=`${r.top}px`;el.style.width=`${r.w}px`;el.style.height=`${r.h}px`;}function onBackdropClick(){close();}function onKeyDown(e){if(e.key==="Escape"&&current)close();}document.addEventListener("keydown",onKeyDown);function open(el,image,index){if(current||!image?.src)return;if(closeTimer){clearTimeout(closeTimer);closeTimer=null;}ensureOverlay();if(!detail||!detailImg||!backdrop)return;const start=relativeRect(el);current=el;returnFocusTo=document.activeElement??null;const dur=o.respectReducedMotion&&prefersReducedMotion()?0:o.openDuration;detail.style.setProperty("--lb-dur",`${dur}ms`);backdrop.style.setProperty("--lb-dur",`${dur}ms`);backdrop.style.background=`rgba(0,0,0,${o.backdropOpacity})`;backdrop.style.backdropFilter=o.backdropBlur?`blur(${o.backdropBlur}px)`:"";// the large candidate belongs here: this is what srcSet is for
applyImage(detailImg,image,Math.round(root.clientWidth*o.openScale));detail.setAttribute("aria-label",image.alt||`Image ${index+1}`);detail.style.borderRadius=`${o.cornerRadius}px`;detail.style.boxShadow=o.openShadow||"none";detail.style.transition="none";applyRect(detail,start);detail.style.opacity="1";// force a reflow, otherwise the browser animates from the end values
void detail.offsetWidth;const ease="cubic-bezier(.22,1,.36,1)";detail.style.transition=["left","top","width","height"].map(p=>`${p} var(--lb-dur) ${ease}`).join(",");applyRect(detail,targetRect(detailImg));backdrop.style.opacity="1";backdrop.style.pointerEvents="auto";detail.style.pointerEvents="auto";detail.focus({preventScroll:true});o.onOpen?.(index,image.src);}function close(animate=true){if(!current||!detail||!backdrop)return;const el=current;backdrop.style.opacity="0";backdrop.style.pointerEvents="none";detail.style.pointerEvents="none";detail.removeAttribute("aria-label");// return focus to where it came from, otherwise a keyboard user
// lands back at the top of the page after closing
returnFocusTo?.focus({preventScroll:true});returnFocusTo=null;if(!animate){detail.style.transition="none";detail.style.opacity="0";current=null;o.onClose?.();return;}// back to where the tile sits now; effects freeze their motion while
// something is open, so that position hasn't shifted
applyRect(detail,relativeRect(el));const dur=o.respectReducedMotion&&prefersReducedMotion()?0:o.openDuration;closeTimer=setTimeout(()=>{if(detail)detail.style.opacity="0";current=null;closeTimer=null;o.onClose?.();},dur);}return{open,close,get openElement(){return current;},update(next){o={...o,...next};},destroy(){if(closeTimer)clearTimeout(closeTimer);document.removeEventListener("keydown",onKeyDown);backdrop?.removeEventListener("click",onBackdropClick);backdrop?.remove();detail?.remove();backdrop=null;detail=null;detailImg=null;current=null;}};}const sphereDefaults={images:[],label:"Rotating image sphere",keyboard:true,count:40,radius:300,autoFit:true,tileWidth:75,tileHeight:94,distance:750,tilt:0,autoRotate:true,axis:"y",speed:14,direction:1,depthFade:.8,hideBack:false,interactive:true,draggable:true,friction:.94,openable:true,...lightboxDefaults,cornerRadius:4,tileShadow:""};const meta={id:"sphere-orbit",number:"002",name:"Sphere Orbit",categories:["drag","click","infinite"],description:"Images spread across a sphere that rotates on its own. Fibonacci distribution, real perspective, drag to spin.",free:true};/**
 * Golden angle. Each successive point offset by it lands as far as possible
 * from all previous ones — that is what makes the distribution even instead
 * of banding into rings or stripes.
 */const GOLDEN_ANGLE=Math.PI*(3-Math.sqrt(5));const DEG=Math.PI/180;/**
 * Container size at which the pixel values in the options read literally.
 *
 * Measured: at 700 the sphere spanned 102–104% of the container's short side
 * and clipped at every size. 900 lands it around 81%, leaving a margin on all
 * four edges. Scaling this constant keeps the density relationship between
 * radius and tile size intact — lowering `radius` alone would not.
 */const FIT_REFERENCE=900;function createSphereOrbit(root,userOptions={}){let o={...sphereDefaults,...userOptions};// Dropped on the canvas with an empty image array the component would
// render nothing, which reads as broken. Neutral placeholders make the
// shape visible immediately.
const withFallback=list=>{const normalized=normalizeImages(list);return normalized.length>0?normalized:placeholderImages();};let pics=withFallback(o.images);let W=0;let H=0;let tiles=[];// orientation as a matrix rather than angles: no pole to seize up at
let R=identity();// angular velocity in camera axes, rad/s — keeps spinning after release
let velYaw=0;let velPitch=0;let spun=0;const pt={x:0,y:0,z:0};let dragging=false;let lastX=0;let lastY=0;let lastT=0;let lastFrame=0;let raf=0;let destroyed=false;const tap=createTapTracker();const tileByEl=new WeakMap;let lightbox=null;let ro=null;let focusEl=null;let stilstaan=o.respectReducedMotion&&prefersReducedMotion();const stopMotionWatch=onReducedMotionChange(reduced=>{stilstaan=o.respectReducedMotion&&reduced;});const prevPosition=root.style.position;if(getComputedStyle(root).position==="static")root.style.position="relative";const prevOverflow=root.style.overflow;const prevTouch=root.style.touchAction;const prevSelect=root.style.userSelect;root.style.overflow="hidden";root.style.touchAction="none";root.style.userSelect="none";root.setAttribute("role","group");root.setAttribute("aria-label",o.label);root.setAttribute("aria-roledescription","rotatable image sphere");function build(){W=root.clientWidth;H=root.clientHeight;for(const t of tiles)t.el.remove();tiles=[];focusEl=null;const n=Math.max(1,Math.floor(o.count));for(let i=0;i<n;i++){// Fibonacci sphere: y runs linearly from 1 to -1 while the angle
// steps by the golden angle. More even than lat/lon, which bunches
// up at the poles.
const uy=n===1?0:1-i/(n-1)*2;const r=Math.sqrt(Math.max(0,1-uy*uy));const theta=i*GOLDEN_ANGLE;const el=document.createElement("div");el.style.cssText="position:absolute;top:0;left:0;overflow:hidden;will-change:transform,opacity;"+`border-radius:${o.cornerRadius}px;box-shadow:${o.tileShadow||"none"}`;const img=document.createElement("img");img.style.cssText="width:100%;height:100%;object-fit:cover;display:block;pointer-events:none";img.draggable=false;img.alt="";img.decoding="async";el.style.cursor=o.openable&&o.interactive?"zoom-in":"";el.setAttribute("role","img");el.appendChild(img);root.appendChild(el);const tile={el,img,index:-1,ux:Math.cos(theta)*r,uy,uz:Math.sin(theta)*r};tileByEl.set(el,tile);tiles.push(tile);}}function layout(){if(tiles.length===0||pics.length===0)return;const cx=W/2;const cy=H/2;// one factor for the whole scene, so the density relationship between
// radius and tile size is preserved at any container size
const fit=o.autoFit?Math.min(W,H)/FIT_REFERENCE:1;const radius=o.radius*fit;const tileW=o.tileWidth*fit;const tileH=o.tileHeight*fit;// the fixed tilt goes on last, in camera axes
const M=o.tilt?multiply(rotationX(o.tilt*DEG),R):R;// the camera must stay outside the sphere, or the projection inverts
const dist=Math.max(o.distance*fit,radius*1.05);for(let i=0;i<tiles.length;i++){const t=tiles[i];transform(M,t.ux,t.uy,t.uz,pt);const z2=pt.z;const wx=pt.x*radius;const wy=pt.y*radius;const wz=pt.z*radius;// perspective divide: scale is exactly 1 at z=0
const scale=dist/(dist-wz);const tw=tileW*scale;const th=tileH*scale;const x=cx+wx*scale-tw/2;const y=cy+wy*scale-th/2;// 0 = fully at the back, 1 = fully at the front
const front=(z2+1)/2;const opacity=o.hideBack?z2>=0?1:0:1-o.depthFade*(1-front);const s=t.el.style;s.width=`${tw}px`;s.height=`${th}px`;// paint in depth order; without this, back tiles poke through the
// front ones
s.zIndex=String(Math.round(front*1e3));s.opacity=t.el===lightbox?.openElement?"0":String(opacity);// no rotation: tiles keep facing the camera
s.transform=`translate3d(${x}px, ${y}px, 0)`;const idx=i%pics.length;if(idx!==t.index){t.index=idx;applyImage(t.img,pics[idx],tw);}else{updateSizes(t.img,tw);}}}function frame(now){if(destroyed)return;const dt=lastFrame?Math.min(.1,(now-lastFrame)/1e3):0;lastFrame=now;if(!dragging&&!lightbox?.openElement){if(o.autoRotate&&!stilstaan){// in object axes: the sphere spins around its own axis, even
// after you have dragged it somewhere else
const a=o.speed*DEG*o.direction*dt;if(o.axis==="x")R=multiply(R,rotationX(a));else if(o.axis==="z")R=multiply(R,rotationZ(a));else if(o.axis==="tumble"){// 0.618 is irrational enough that the path never repeats
R=multiply(multiply(R,rotationY(a)),rotationX(a*.618));}else R=multiply(R,rotationY(a));}// coasting happens in camera axes, same as dragging
if(velYaw||velPitch){R=multiply(multiply(rotationY(velYaw*dt),rotationX(velPitch*dt)),R);const decay=Math.pow(o.friction,dt*60);velYaw*=decay;velPitch*=decay;if(Math.abs(velYaw)<5e-4)velYaw=0;if(Math.abs(velPitch)<5e-4)velPitch=0;}// rounding error accumulates; straighten it out now and then
if(++spun%240===0)R=orthonormalize(R);}layout();markFocus();raf=requestAnimationFrame(frame);}const onDown=e=>{if(lightbox?.openElement)return;// state first, capture second: setPointerCapture can throw (pointer
// already released, or a synthetic event from a test) and then the
// rest of this handler never runs
lastX=e.clientX;lastY=e.clientY;lastT=performance.now();velYaw=0;velPitch=0;if(o.draggable){dragging=true;root.style.cursor="grabbing";try{root.setPointerCapture(e.pointerId);}catch{// dragging still works without capture, just not outside the element
}}// capture on pointerdown: setPointerCapture retargets pointerup
const hit=e.target?.closest?.("div");tap.down(e,hit?tileByEl.get(hit)??null:null);};const onMove=e=>{tap.move(e);if(!dragging)return;const now=performance.now();const dx=e.clientX-lastX;const dy=e.clientY-lastY;const dt=Math.max(1,now-lastT)/1e3;// a drag across the full width is roughly half a revolution
const perPx=Math.PI/Math.max(1,W);const ay=dx*perPx;const ax=dy*perPx;// camera axes applied before the existing orientation: that is
// trackball behaviour, and it works at any angle without a limit
R=multiply(multiply(rotationY(ay),rotationX(ax)),R);velYaw=ay/dt;velPitch=ax/dt;lastX=e.clientX;lastY=e.clientY;lastT=now;};const onUp=()=>{if(dragging){dragging=false;root.style.cursor=o.draggable?"grab":"";}const tile=tap.up();if(!tile||!o.openable||lightbox?.openElement)return;if(tile.index<0||!pics[tile.index])return;// the back side is not clickable
if(parseFloat(tile.el.style.opacity||"0")<.35)return;velYaw=0;velPitch=0;lightbox?.open(tile.el,pics[tile.index],tile.index);};/** The frontmost tile — that is what Enter opens. */function frontTile(){let beste=null;let besteZ=-Infinity;for(const t of tiles){if(t.index<0)continue;const z=+(t.el.style.zIndex||0);if(z>besteZ){besteZ=z;beste=t;}}return beste;}/**
     * Focus state is read, not tracked through focus/blur events. Those don't
     * fire in every situation — for instance when the window itself isn't
     * focused — and then a tracked flag goes stale.
     */function markFocus(){const target=document.activeElement===root?frontTile():null;const el=target?.el??null;// only the two elements that actually change; this runs every frame
if(el===focusEl)return;if(focusEl)focusEl.style.boxShadow=composeShadow(o.tileShadow,false);if(el)el.style.boxShadow=composeShadow(o.tileShadow,true);focusEl=el;}/** Reapply the shadow everywhere, after the option changed. */function refreshShadows(){for(const t of tiles){t.el.style.boxShadow=composeShadow(o.tileShadow,t.el===focusEl);}}const onKey=e=>{if(!o.keyboard||lightbox?.openElement)return;const stap=(e.shiftKey?24:8)*DEG;switch(e.key){case"ArrowLeft":R=multiply(rotationY(-stap),R);break;case"ArrowRight":R=multiply(rotationY(stap),R);break;case"ArrowUp":R=multiply(rotationX(-stap),R);break;case"ArrowDown":R=multiply(rotationX(stap),R);break;case"Enter":case" ":{if(!o.openable)return;const t=frontTile();if(!t||t.index<0||!pics[t.index])return;velYaw=0;velPitch=0;lightbox?.open(t.el,pics[t.index],t.index);break;}default:return;}e.preventDefault();velYaw=0;velPitch=0;};function attach(){root.style.cursor=o.draggable?"grab":"";root.tabIndex=0;root.addEventListener("keydown",onKey);root.addEventListener("pointerdown",onDown);root.addEventListener("pointermove",onMove);root.addEventListener("pointerup",onUp);root.addEventListener("pointercancel",onUp);}function detach(){root.style.cursor="";root.removeAttribute("tabindex");root.removeEventListener("keydown",onKey);root.removeEventListener("pointerdown",onDown);root.removeEventListener("pointermove",onMove);root.removeEventListener("pointerup",onUp);root.removeEventListener("pointercancel",onUp);}function start(){if(o.interactive){attach();if(o.openable&&!lightbox)lightbox=createLightbox(root,o);lastFrame=0;raf=requestAnimationFrame(frame);}else{layout();}}build();start();ro=new ResizeObserver(()=>{lightbox?.close(false);W=root.clientWidth;H=root.clientHeight;if(!o.interactive)layout();});ro.observe(root);return{update(next){const needsRebuild=next.count!==undefined&&next.count!==o.count||next.cornerRadius!==undefined&&next.cornerRadius!==o.cornerRadius;const wasInteractive=o.interactive;const imagesChanged=next.images!==undefined&&next.images!==o.images;const shadowChanged=next.tileShadow!==undefined&&next.tileShadow!==o.tileShadow;o={...o,...next};if(shadowChanged)refreshShadows();if(imagesChanged){pics=withFallback(o.images);// the stored index refers to the previous list; leaving it in
// place makes layout() skip tiles whose index happens to match
for(const t of tiles)t.index=-1;}lightbox?.update(o);if(needsRebuild){lightbox?.close(false);build();}if(o.interactive!==wasInteractive){lightbox?.close(false);cancelAnimationFrame(raf);detach();start();}else if(!o.interactive){layout();}},resize(){lightbox?.close(false);W=root.clientWidth;H=root.clientHeight;if(!o.interactive)layout();},destroy(){destroyed=true;cancelAnimationFrame(raf);ro?.disconnect();detach();lightbox?.destroy();lightbox=null;for(const t of tiles)t.el.remove();tiles=[];stopMotionWatch();root.style.position=prevPosition;root.style.overflow=prevOverflow;root.style.touchAction=prevTouch;root.style.userSelect=prevSelect;root.style.cursor="";root.removeAttribute("tabindex");root.removeAttribute("role");root.removeAttribute("aria-label");root.removeAttribute("aria-roledescription");}};}// ──────────────────────────────────────────────────── framer adapter
/**
 * Sphere Orbit
 *
 * @framerIntrinsicWidth 900
 * @framerIntrinsicHeight 700
 * @framerSupportedLayoutWidth any
 * @framerSupportedLayoutHeight any
 */export default function SphereOrbit(props){const{images,background,style,...options}=props;const ref=useRef(null);const instance=useRef(null);// No rAF loop and no pointer handlers in a static context. RenderTarget
// .canvas alone was not enough: it misses export and thumbnail, where the
// animation kept running and produced artifacts in exported assets.
const interactive=!useIsStaticRenderer();// pass the whole ResponsiveImage, not just .src — otherwise we lose
// srcSet (a sharp image when opened) and alt (screen readers)
const pics=normalizeImages(images);// only rebuild when the images, the count or the radius change
const rebuildKey=`${imagesKey(pics)}|${options.count}|${options.cornerRadius}|${interactive}`;useEffect(()=>{const el=ref.current;// no early return on an empty list: the effect falls back to
// placeholder cards itself, and bailing out here is what made the
// component render nothing at all in Framer
if(!el)return;instance.current=createSphereOrbit(el,{...options,images:pics,interactive});return()=>{instance.current?.destroy();instance.current=null;};},[rebuildKey]);// dragging a slider goes through update(), without a rebuild
useEffect(()=>{instance.current?.update({...options,images:pics,interactive});});return /*#__PURE__*/_jsx("div",{ref:ref,style:{...style,position:"relative",overflow:"hidden",background}});}SphereOrbit.defaultProps={images:[],background:"#0d0d0f",label:"Rotating image sphere",keyboard:true,respectReducedMotion:true,count:40,radius:300,autoFit:true,tileWidth:75,tileHeight:94,distance:750,tilt:0,autoRotate:true,axis:"y",speed:14,direction:1,depthFade:.8,hideBack:false,cornerRadius:4,tileShadow:"",openShadow:"0px 30px 80px 0px rgba(0,0,0,0.6)",draggable:true,friction:.94,openable:true,openScale:.8,openDuration:520,backdropOpacity:.72,backdropBlur:10};