(globalThis.TURBOPACK || (globalThis.TURBOPACK = [])).push([typeof document === "object" ? document.currentScript : undefined,
"[project]/src/components/ColorBends.tsx [app-client] (ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "default",
    ()=>ColorBends
]);
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/next/dist/compiled/react/jsx-dev-runtime.js [app-client] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$compiler$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/next/dist/compiled/react/compiler-runtime.js [app-client] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/next/dist/compiled/react/index.js [app-client] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$three$2f$build$2f$three$2e$module$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/three/build/three.module.js [app-client] (ecmascript)");
;
var _s = __turbopack_context__.k.signature();
'use client';
;
;
;
const MAX_COLORS = 8;
const frag = `
#define MAX_COLORS ${MAX_COLORS}
uniform vec2 uCanvas;
uniform float uTime;
uniform float uSpeed;
uniform vec2 uRot;
uniform int uColorCount;
uniform vec3 uColors[MAX_COLORS];
uniform int uTransparent;
uniform float uScale;
uniform float uFrequency;
uniform float uWarpStrength;
uniform vec2 uPointer; // in NDC [-1,1]
uniform float uMouseInfluence;
uniform float uParallax;
uniform float uNoise;
varying vec2 vUv;

void main() {
  float t = uTime * uSpeed;
  vec2 p = vUv * 2.0 - 1.0;
  p += uPointer * uParallax * 0.1;
  vec2 rp = vec2(p.x * uRot.x - p.y * uRot.y, p.x * uRot.y + p.y * uRot.x);
  vec2 q = vec2(rp.x * (uCanvas.x / uCanvas.y), rp.y);
  q /= max(uScale, 0.0001);
  q /= 0.5 + 0.2 * dot(q, q);
  q += 0.2 * cos(t) - 7.56;
  vec2 toward = (uPointer - rp);
  q += toward * uMouseInfluence * 0.2;

    vec3 col = vec3(0.0);
    float a = 1.0;

    if (uColorCount > 0) {
      vec2 s = q;
      vec3 sumCol = vec3(0.0);
      float cover = 0.0;
      for (int i = 0; i < MAX_COLORS; ++i) {
            if (i >= uColorCount) break;
            s -= 0.01;
            vec2 r = sin(1.5 * (s.yx * uFrequency) + 2.0 * cos(s * uFrequency));
            float m0 = length(r + sin(5.0 * r.y * uFrequency - 3.0 * t + float(i)) / 4.0);
            float kBelow = clamp(uWarpStrength, 0.0, 1.0);
            float kMix = pow(kBelow, 0.3); // strong response across 0..1
            float gain = 1.0 + max(uWarpStrength - 1.0, 0.0); // allow >1 to amplify displacement
            vec2 disp = (r - s) * kBelow;
            vec2 warped = s + disp * gain;
            float m1 = length(warped + sin(5.0 * warped.y * uFrequency - 3.0 * t + float(i)) / 4.0);
            float m = mix(m0, m1, kMix);
            float w = 1.0 - exp(-6.0 / exp(6.0 * m));
            sumCol += uColors[i] * w;
            cover = max(cover, w);
      }
      col = clamp(sumCol, 0.0, 1.0);
      a = uTransparent > 0 ? cover : 1.0;
    } else {
        vec2 s = q;
        for (int k = 0; k < 3; ++k) {
            s -= 0.01;
            vec2 r = sin(1.5 * (s.yx * uFrequency) + 2.0 * cos(s * uFrequency));
            float m0 = length(r + sin(5.0 * r.y * uFrequency - 3.0 * t + float(k)) / 4.0);
            float kBelow = clamp(uWarpStrength, 0.0, 1.0);
            float kMix = pow(kBelow, 0.3);
            float gain = 1.0 + max(uWarpStrength - 1.0, 0.0);
            vec2 disp = (r - s) * kBelow;
            vec2 warped = s + disp * gain;
            float m1 = length(warped + sin(5.0 * warped.y * uFrequency - 3.0 * t + float(k)) / 4.0);
            float m = mix(m0, m1, kMix);
            col[k] = 1.0 - exp(-6.0 / exp(6.0 * m));
        }
        a = uTransparent > 0 ? max(max(col.r, col.g), col.b) : 1.0;
    }

    if (uNoise > 0.0001) {
      float n = fract(sin(dot(gl_FragCoord.xy + vec2(uTime), vec2(12.9898, 78.233))) * 43758.5453123);
      col += (n - 0.5) * uNoise;
      col = clamp(col, 0.0, 1.0);
    }

    vec3 rgb = (uTransparent > 0) ? col * a : col;
    gl_FragColor = vec4(rgb, a);
}
`;
const vert = `
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = vec4(position, 1.0);
}
`;
function ColorBends(t0) {
    _s();
    const $ = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$compiler$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["c"])(33);
    if ($[0] !== "3825730de74d1e102cc16f22355aa4b6d123943dba0e0b028cccff8f7c48d527") {
        for(let $i = 0; $i < 33; $i += 1){
            $[$i] = Symbol.for("react.memo_cache_sentinel");
        }
        $[0] = "3825730de74d1e102cc16f22355aa4b6d123943dba0e0b028cccff8f7c48d527";
    }
    const { className, style, rotation: t1, speed: t2, colors: t3, transparent: t4, autoRotate: t5, scale: t6, frequency: t7, warpStrength: t8, mouseInfluence: t9, parallax: t10, noise: t11 } = t0;
    const rotation = t1 === undefined ? 45 : t1;
    const speed = t2 === undefined ? 0.2 : t2;
    let t12;
    if ($[1] !== t3) {
        t12 = t3 === undefined ? [] : t3;
        $[1] = t3;
        $[2] = t12;
    } else {
        t12 = $[2];
    }
    const colors = t12;
    const transparent = t4 === undefined ? true : t4;
    const autoRotate = t5 === undefined ? 0 : t5;
    const scale = t6 === undefined ? 1 : t6;
    const frequency = t7 === undefined ? 1 : t7;
    const warpStrength = t8 === undefined ? 1 : t8;
    const mouseInfluence = t9 === undefined ? 1 : t9;
    const parallax = t10 === undefined ? 0.5 : t10;
    const noise = t11 === undefined ? 0.1 : t11;
    const containerRef = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useRef"])(null);
    const rendererRef = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useRef"])(null);
    const rafRef = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useRef"])(null);
    const materialRef = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useRef"])(null);
    const resizeObserverRef = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useRef"])(null);
    const rotationRef = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useRef"])(rotation);
    const autoRotateRef = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useRef"])(autoRotate);
    let t13;
    if ($[3] === Symbol.for("react.memo_cache_sentinel")) {
        t13 = new __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$three$2f$build$2f$three$2e$module$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__.Vector2(0, 0);
        $[3] = t13;
    } else {
        t13 = $[3];
    }
    const pointerTargetRef = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useRef"])(t13);
    let t14;
    if ($[4] === Symbol.for("react.memo_cache_sentinel")) {
        t14 = new __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$three$2f$build$2f$three$2e$module$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__.Vector2(0, 0);
        $[4] = t14;
    } else {
        t14 = $[4];
    }
    const pointerCurrentRef = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useRef"])(t14);
    const pointerSmoothRef = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useRef"])(8);
    let t15;
    if ($[5] !== frequency || $[6] !== mouseInfluence || $[7] !== noise || $[8] !== parallax || $[9] !== scale || $[10] !== speed || $[11] !== transparent || $[12] !== warpStrength) {
        t15 = ({
            "ColorBends[useEffect()]": ()=>{
                const container = containerRef.current;
                const scene = new __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$three$2f$build$2f$three$2e$module$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__.Scene();
                const camera = new __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$three$2f$build$2f$three$2e$module$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__.OrthographicCamera(-1, 1, 1, -1, 0, 1);
                const geometry = new __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$three$2f$build$2f$three$2e$module$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__.PlaneGeometry(2, 2);
                const uColorsArray = Array.from({
                    length: MAX_COLORS
                }, _ColorBendsUseEffectArrayFrom);
                const material = new __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$three$2f$build$2f$three$2e$module$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__.ShaderMaterial({
                    vertexShader: vert,
                    fragmentShader: frag,
                    uniforms: {
                        uCanvas: {
                            value: new __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$three$2f$build$2f$three$2e$module$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__.Vector2(1, 1)
                        },
                        uTime: {
                            value: 0
                        },
                        uSpeed: {
                            value: speed
                        },
                        uRot: {
                            value: new __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$three$2f$build$2f$three$2e$module$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__.Vector2(1, 0)
                        },
                        uColorCount: {
                            value: 0
                        },
                        uColors: {
                            value: uColorsArray
                        },
                        uTransparent: {
                            value: transparent ? 1 : 0
                        },
                        uScale: {
                            value: scale
                        },
                        uFrequency: {
                            value: frequency
                        },
                        uWarpStrength: {
                            value: warpStrength
                        },
                        uPointer: {
                            value: new __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$three$2f$build$2f$three$2e$module$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__.Vector2(0, 0)
                        },
                        uMouseInfluence: {
                            value: mouseInfluence
                        },
                        uParallax: {
                            value: parallax
                        },
                        uNoise: {
                            value: noise
                        }
                    },
                    premultipliedAlpha: true,
                    transparent: true
                });
                materialRef.current = material;
                const mesh = new __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$three$2f$build$2f$three$2e$module$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__.Mesh(geometry, material);
                scene.add(mesh);
                const renderer = new __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$three$2f$build$2f$three$2e$module$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__.WebGLRenderer({
                    antialias: false,
                    powerPreference: "high-performance",
                    alpha: true
                });
                rendererRef.current = renderer;
                renderer.outputColorSpace = __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$three$2f$build$2f$three$2e$module$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__.SRGBColorSpace;
                renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
                renderer.setClearColor(0, transparent ? 0 : 1);
                renderer.domElement.style.width = "100%";
                renderer.domElement.style.height = "100%";
                renderer.domElement.style.display = "block";
                container.appendChild(renderer.domElement);
                const clock = new __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$three$2f$build$2f$three$2e$module$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__.Clock();
                const handleResize = {
                    "ColorBends[useEffect() > handleResize]": ()=>{
                        const w = container.clientWidth || 1;
                        const h = container.clientHeight || 1;
                        renderer.setSize(w, h, false);
                        material.uniforms.uCanvas.value.set(w, h);
                    }
                }["ColorBends[useEffect() > handleResize]"];
                handleResize();
                if ("ResizeObserver" in window) {
                    const ro = new ResizeObserver(handleResize);
                    ro.observe(container);
                    resizeObserverRef.current = ro;
                } else {
                    window.addEventListener("resize", handleResize);
                }
                const loop = {
                    "ColorBends[useEffect() > loop]": ()=>{
                        const dt = clock.getDelta();
                        const elapsed = clock.elapsedTime;
                        material.uniforms.uTime.value = elapsed;
                        const deg = rotationRef.current % 360 + autoRotateRef.current * elapsed;
                        const rad = deg * Math.PI / 180;
                        const c = Math.cos(rad);
                        const s = Math.sin(rad);
                        material.uniforms.uRot.value.set(c, s);
                        const cur = pointerCurrentRef.current;
                        const tgt = pointerTargetRef.current;
                        const amt = Math.min(1, dt * pointerSmoothRef.current);
                        cur.lerp(tgt, amt);
                        material.uniforms.uPointer.value.copy(cur);
                        renderer.render(scene, camera);
                        rafRef.current = requestAnimationFrame(loop);
                    }
                }["ColorBends[useEffect() > loop]"];
                rafRef.current = requestAnimationFrame(loop);
                return ()=>{
                    if (rafRef.current !== null) {
                        cancelAnimationFrame(rafRef.current);
                    }
                    if (resizeObserverRef.current) {
                        resizeObserverRef.current.disconnect();
                    } else {
                        window.removeEventListener("resize", handleResize);
                    }
                    geometry.dispose();
                    material.dispose();
                    renderer.dispose();
                    if (renderer.domElement && renderer.domElement.parentElement === container) {
                        container.removeChild(renderer.domElement);
                    }
                };
            }
        })["ColorBends[useEffect()]"];
        $[5] = frequency;
        $[6] = mouseInfluence;
        $[7] = noise;
        $[8] = parallax;
        $[9] = scale;
        $[10] = speed;
        $[11] = transparent;
        $[12] = warpStrength;
        $[13] = t15;
    } else {
        t15 = $[13];
    }
    let t16;
    if ($[14] === Symbol.for("react.memo_cache_sentinel")) {
        t16 = [];
        $[14] = t16;
    } else {
        t16 = $[14];
    }
    (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useEffect"])(t15, t16);
    let t17;
    let t18;
    if ($[15] !== autoRotate || $[16] !== colors || $[17] !== frequency || $[18] !== mouseInfluence || $[19] !== noise || $[20] !== parallax || $[21] !== rotation || $[22] !== scale || $[23] !== speed || $[24] !== transparent || $[25] !== warpStrength) {
        t17 = ({
            "ColorBends[useEffect()]": ()=>{
                const material_0 = materialRef.current;
                const renderer_0 = rendererRef.current;
                if (!material_0) {
                    return;
                }
                rotationRef.current = rotation;
                autoRotateRef.current = autoRotate;
                material_0.uniforms.uSpeed.value = speed;
                material_0.uniforms.uScale.value = scale;
                material_0.uniforms.uFrequency.value = frequency;
                material_0.uniforms.uWarpStrength.value = warpStrength;
                material_0.uniforms.uMouseInfluence.value = mouseInfluence;
                material_0.uniforms.uParallax.value = parallax;
                material_0.uniforms.uNoise.value = noise;
                const toVec3 = _ColorBendsUseEffectToVec;
                const arr = (colors || []).filter(Boolean).slice(0, MAX_COLORS).map(toVec3);
                for(let i = 0; i < MAX_COLORS; i++){
                    const vec = material_0.uniforms.uColors.value[i];
                    if (i < arr.length) {
                        vec.copy(arr[i]);
                    } else {
                        vec.set(0, 0, 0);
                    }
                }
                material_0.uniforms.uColorCount.value = arr.length;
                material_0.uniforms.uTransparent.value = transparent ? 1 : 0;
                if (renderer_0) {
                    renderer_0.setClearColor(0, transparent ? 0 : 1);
                }
            }
        })["ColorBends[useEffect()]"];
        t18 = [
            rotation,
            autoRotate,
            speed,
            scale,
            frequency,
            warpStrength,
            mouseInfluence,
            parallax,
            noise,
            colors,
            transparent
        ];
        $[15] = autoRotate;
        $[16] = colors;
        $[17] = frequency;
        $[18] = mouseInfluence;
        $[19] = noise;
        $[20] = parallax;
        $[21] = rotation;
        $[22] = scale;
        $[23] = speed;
        $[24] = transparent;
        $[25] = warpStrength;
        $[26] = t17;
        $[27] = t18;
    } else {
        t17 = $[26];
        t18 = $[27];
    }
    (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useEffect"])(t17, t18);
    let t19;
    let t20;
    if ($[28] === Symbol.for("react.memo_cache_sentinel")) {
        t19 = ({
            "ColorBends[useEffect()]": ()=>{
                const material_1 = materialRef.current;
                const container_0 = containerRef.current;
                if (!material_1 || !container_0) {
                    return;
                }
                const handlePointerMove = {
                    "ColorBends[useEffect() > handlePointerMove]": (e)=>{
                        const rect = container_0.getBoundingClientRect();
                        const x = (e.clientX - rect.left) / (rect.width || 1) * 2 - 1;
                        const y = -((e.clientY - rect.top) / (rect.height || 1) * 2 - 1);
                        pointerTargetRef.current.set(x, y);
                    }
                }["ColorBends[useEffect() > handlePointerMove]"];
                container_0.addEventListener("pointermove", handlePointerMove);
                return ()=>{
                    container_0.removeEventListener("pointermove", handlePointerMove);
                };
            }
        })["ColorBends[useEffect()]"];
        t20 = [];
        $[28] = t19;
        $[29] = t20;
    } else {
        t19 = $[28];
        t20 = $[29];
    }
    (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useEffect"])(t19, t20);
    const t21 = `w-full h-full relative overflow-hidden ${className}`;
    let t22;
    if ($[30] !== style || $[31] !== t21) {
        t22 = /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
            ref: containerRef,
            className: t21,
            style: style
        }, void 0, false, {
            fileName: "[project]/src/components/ColorBends.tsx",
            lineNumber: 420,
            columnNumber: 11
        }, this);
        $[30] = style;
        $[31] = t21;
        $[32] = t22;
    } else {
        t22 = $[32];
    }
    return t22;
}
_s(ColorBends, "1RRNKepfqTAyKhfYAEjyJC34N9c=");
_c = ColorBends;
function _ColorBendsUseEffectToVec(hex) {
    const h_0 = hex.replace("#", "").trim();
    const v = h_0.length === 3 ? [
        parseInt(h_0[0] + h_0[0], 16),
        parseInt(h_0[1] + h_0[1], 16),
        parseInt(h_0[2] + h_0[2], 16)
    ] : [
        parseInt(h_0.slice(0, 2), 16),
        parseInt(h_0.slice(2, 4), 16),
        parseInt(h_0.slice(4, 6), 16)
    ];
    return new __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$three$2f$build$2f$three$2e$module$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__.Vector3(v[0] / 255, v[1] / 255, v[2] / 255);
}
function _ColorBendsUseEffectArrayFrom() {
    return new __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$three$2f$build$2f$three$2e$module$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__.Vector3(0, 0, 0);
}
var _c;
__turbopack_context__.k.register(_c, "ColorBends");
if (typeof globalThis.$RefreshHelpers$ === 'object' && globalThis.$RefreshHelpers !== null) {
    __turbopack_context__.k.registerExports(__turbopack_context__.m, globalThis.$RefreshHelpers$);
}
}),
"[project]/src/components/BlurText.tsx [app-client] (ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "default",
    ()=>__TURBOPACK__default__export__
]);
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/next/dist/compiled/react/jsx-dev-runtime.js [app-client] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$framer$2d$motion$2f$dist$2f$es$2f$render$2f$components$2f$motion$2f$proxy$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/framer-motion/dist/es/render/components/motion/proxy.mjs [app-client] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/next/dist/compiled/react/index.js [app-client] (ecmascript)");
;
var _s = __turbopack_context__.k.signature();
"use client";
;
;
const buildKeyframes = (from, steps)=>{
    const keys = new Set([
        ...Object.keys(from),
        ...steps.flatMap((s)=>Object.keys(s))
    ]);
    const keyframes = {};
    keys.forEach((k)=>{
        keyframes[k] = [
            from[k],
            ...steps.map((s)=>s[k])
        ];
    });
    return keyframes;
};
const BlurText = ({ text = '', delay = 200, className = '', animateBy = 'words', direction = 'top', threshold = 0.1, rootMargin = '0px', animationFrom, animationTo, easing = (t)=>t, onAnimationComplete, stepDuration = 0.35 })=>{
    _s();
    const elements = animateBy === 'words' ? text.split(' ') : text.split('');
    const [inView, setInView] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])(false);
    const ref = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useRef"])(null);
    (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useEffect"])({
        "BlurText.useEffect": ()=>{
            if (!ref.current) return;
            const observer = new IntersectionObserver({
                "BlurText.useEffect": ([entry])=>{
                    if (entry.isIntersecting) {
                        setInView(true);
                        observer.unobserve(ref.current);
                    }
                }
            }["BlurText.useEffect"], {
                threshold,
                rootMargin
            });
            observer.observe(ref.current);
            return ({
                "BlurText.useEffect": ()=>observer.disconnect()
            })["BlurText.useEffect"];
        }
    }["BlurText.useEffect"], [
        threshold,
        rootMargin
    ]);
    const defaultFrom = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useMemo"])({
        "BlurText.useMemo[defaultFrom]": ()=>direction === 'top' ? {
                filter: 'blur(10px)',
                opacity: 0,
                y: -50
            } : {
                filter: 'blur(10px)',
                opacity: 0,
                y: 50
            }
    }["BlurText.useMemo[defaultFrom]"], [
        direction
    ]);
    const defaultTo = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useMemo"])({
        "BlurText.useMemo[defaultTo]": ()=>[
                {
                    filter: 'blur(5px)',
                    opacity: 0.5,
                    y: direction === 'top' ? 5 : -5
                },
                {
                    filter: 'blur(0px)',
                    opacity: 1,
                    y: 0
                }
            ]
    }["BlurText.useMemo[defaultTo]"], [
        direction
    ]);
    const fromSnapshot = animationFrom ?? defaultFrom;
    const toSnapshots = animationTo ?? defaultTo;
    const stepCount = toSnapshots.length + 1;
    const totalDuration = stepDuration * (stepCount - 1);
    const times = Array.from({
        length: stepCount
    }, (_, i)=>stepCount === 1 ? 0 : i / (stepCount - 1));
    return /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("p", {
        ref: ref,
        className: `blur-text ${className} flex flex-wrap`,
        children: elements.map((segment, index)=>{
            const animateKeyframes = buildKeyframes(fromSnapshot, toSnapshots);
            const spanTransition = {
                duration: totalDuration,
                times,
                delay: index * delay / 1000,
                ease: easing
            };
            return /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$framer$2d$motion$2f$dist$2f$es$2f$render$2f$components$2f$motion$2f$proxy$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["motion"].span, {
                initial: fromSnapshot,
                animate: inView ? animateKeyframes : fromSnapshot,
                transition: spanTransition,
                onAnimationComplete: index === elements.length - 1 ? onAnimationComplete : undefined,
                style: {
                    display: 'inline-block',
                    willChange: 'transform, filter, opacity'
                },
                children: [
                    segment === ' ' ? '\u00A0' : segment,
                    animateBy === 'words' && index < elements.length - 1 && '\u00A0'
                ]
            }, index, true, {
                fileName: "[project]/src/components/BlurText.tsx",
                lineNumber: 92,
                columnNumber: 14
            }, ("TURBOPACK compile-time value", void 0));
        })
    }, void 0, false, {
        fileName: "[project]/src/components/BlurText.tsx",
        lineNumber: 83,
        columnNumber: 10
    }, ("TURBOPACK compile-time value", void 0));
};
_s(BlurText, "IxnOfDOifNR2DM+0TATKkPR86iM=");
_c = BlurText;
const __TURBOPACK__default__export__ = BlurText;
var _c;
__turbopack_context__.k.register(_c, "BlurText");
if (typeof globalThis.$RefreshHelpers$ === 'object' && globalThis.$RefreshHelpers !== null) {
    __turbopack_context__.k.registerExports(__turbopack_context__.m, globalThis.$RefreshHelpers$);
}
}),
"[project]/src/components/BubbleMenu.tsx [app-client] (ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "default",
    ()=>BubbleMenu
]);
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/next/dist/compiled/react/jsx-dev-runtime.js [app-client] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$compiler$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/next/dist/compiled/react/compiler-runtime.js [app-client] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/next/dist/compiled/react/index.js [app-client] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$gsap$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$locals$3e$__ = __turbopack_context__.i("[project]/node_modules/gsap/index.js [app-client] (ecmascript) <locals>");
;
var _s = __turbopack_context__.k.signature();
"use client";
;
;
;
function BubbleMenu(t0) {
    _s();
    const $ = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$compiler$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["c"])(23);
    if ($[0] !== "966708f30b0016b1b76843461b6d66b61df1bcb9fd9b4e93cf42034fc8abaf80") {
        for(let $i = 0; $i < 23; $i += 1){
            $[$i] = Symbol.for("react.memo_cache_sentinel");
        }
        $[0] = "966708f30b0016b1b76843461b6d66b61df1bcb9fd9b4e93cf42034fc8abaf80";
    }
    const { items: t1, className, style, menuBg: t2, menuContentColor: t3, animationDuration: t4, staggerDelay: t5 } = t0;
    let t6;
    if ($[1] !== t1) {
        t6 = t1 === undefined ? [] : t1;
        $[1] = t1;
        $[2] = t6;
    } else {
        t6 = $[2];
    }
    const items = t6;
    const menuBg = t2 === undefined ? "#fff" : t2;
    const menuContentColor = t3 === undefined ? "#111" : t3;
    const animationDuration = t4 === undefined ? 0.6 : t4;
    const staggerDelay = t5 === undefined ? 0.15 : t5;
    const containerRef = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useRef"])(null);
    let t7;
    if ($[3] !== animationDuration || $[4] !== staggerDelay) {
        t7 = ({
            "BubbleMenu[useLayoutEffect()]": ()=>{
                const ctx = __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$gsap$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$locals$3e$__["gsap"].context({
                    "BubbleMenu[useLayoutEffect() > gsap.context()]": ()=>{
                        const container = containerRef.current;
                        if (!container) {
                            return;
                        }
                        const bubbles = Array.from(container.querySelectorAll(".bubble-item"));
                        if (!bubbles.length) {
                            return;
                        }
                        __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$gsap$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$locals$3e$__["gsap"].killTweensOf(bubbles);
                        __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$gsap$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$locals$3e$__["gsap"].set(bubbles, {
                            clearProps: "transform"
                        });
                        __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$gsap$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$locals$3e$__["gsap"].set(bubbles, {
                            scale: 0,
                            transformOrigin: "50% 50%",
                            rotate: _temp,
                            force3D: true
                        });
                        __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$gsap$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$locals$3e$__["gsap"].to(bubbles, {
                            scale: 1,
                            duration: animationDuration,
                            ease: "back.out(1.7)",
                            stagger: staggerDelay
                        });
                    }
                }["BubbleMenu[useLayoutEffect() > gsap.context()]"], containerRef);
                return ()=>ctx.revert();
            }
        })["BubbleMenu[useLayoutEffect()]"];
        $[3] = animationDuration;
        $[4] = staggerDelay;
        $[5] = t7;
    } else {
        t7 = $[5];
    }
    let t8;
    if ($[6] !== animationDuration || $[7] !== items || $[8] !== staggerDelay) {
        t8 = [
            items,
            animationDuration,
            staggerDelay
        ];
        $[6] = animationDuration;
        $[7] = items;
        $[8] = staggerDelay;
        $[9] = t8;
    } else {
        t8 = $[9];
    }
    (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useLayoutEffect"])(t7, t8);
    let t9;
    if ($[10] !== className) {
        t9 = [
            "bubble-inline-menu flex justify-center items-center flex-wrap gap-3",
            className
        ].filter(Boolean);
        $[10] = className;
        $[11] = t9;
    } else {
        t9 = $[11];
    }
    const t10 = t9.join(" ");
    let t11;
    if ($[12] !== items || $[13] !== menuBg || $[14] !== menuContentColor) {
        let t12;
        if ($[16] !== menuBg || $[17] !== menuContentColor) {
            t12 = ({
                "BubbleMenu[items.map()]": (item, idx)=>/*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("a", {
                        href: item.href,
                        "aria-label": item.ariaLabel || item.label,
                        className: "bubble-item rounded-full px-6 py-3 text-lg md:text-xl font-semibold transition-[background,color,box-shadow] duration-300 shadow-md hover:shadow-lg will-change-transform",
                        "data-rotate": item.rotation ?? 0,
                        style: {
                            background: menuBg,
                            color: menuContentColor
                        },
                        onMouseEnter: {
                            "BubbleMenu[items.map() > <a>.onMouseEnter]": (e)=>{
                                const el_0 = e.currentTarget;
                                el_0.style.background = item.hoverStyles?.bgColor || "#ddd";
                                el_0.style.color = item.hoverStyles?.textColor || menuContentColor;
                            }
                        }["BubbleMenu[items.map() > <a>.onMouseEnter]"],
                        onMouseLeave: {
                            "BubbleMenu[items.map() > <a>.onMouseLeave]": (e_0)=>{
                                const el_1 = e_0.currentTarget;
                                el_1.style.background = menuBg;
                                el_1.style.color = menuContentColor;
                            }
                        }["BubbleMenu[items.map() > <a>.onMouseLeave]"],
                        children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                            className: "bubble-label inline-block",
                            children: item.label
                        }, void 0, false, {
                            fileName: "[project]/src/components/BubbleMenu.tsx",
                            lineNumber: 138,
                            columnNumber: 58
                        }, this)
                    }, idx, false, {
                        fileName: "[project]/src/components/BubbleMenu.tsx",
                        lineNumber: 123,
                        columnNumber: 51
                    }, this)
            })["BubbleMenu[items.map()]"];
            $[16] = menuBg;
            $[17] = menuContentColor;
            $[18] = t12;
        } else {
            t12 = $[18];
        }
        t11 = items.map(t12);
        $[12] = items;
        $[13] = menuBg;
        $[14] = menuContentColor;
        $[15] = t11;
    } else {
        t11 = $[15];
    }
    let t12;
    if ($[19] !== style || $[20] !== t10 || $[21] !== t11) {
        t12 = /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
            ref: containerRef,
            className: t10,
            style: style,
            children: t11
        }, void 0, false, {
            fileName: "[project]/src/components/BubbleMenu.tsx",
            lineNumber: 156,
            columnNumber: 11
        }, this);
        $[19] = style;
        $[20] = t10;
        $[21] = t11;
        $[22] = t12;
    } else {
        t12 = $[22];
    }
    return t12;
}
_s(BubbleMenu, "JVErPvg7bZ6yLj50J4lCvDO7Tjk=");
_c = BubbleMenu;
function _temp(_, el) {
    return Number(el.getAttribute("data-rotate")) || 0;
}
var _c;
__turbopack_context__.k.register(_c, "BubbleMenu");
if (typeof globalThis.$RefreshHelpers$ === 'object' && globalThis.$RefreshHelpers !== null) {
    __turbopack_context__.k.registerExports(__turbopack_context__.m, globalThis.$RefreshHelpers$);
}
}),
"[project]/src/components/ClickSpark.tsx [app-client] (ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "default",
    ()=>__TURBOPACK__default__export__
]);
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/next/dist/compiled/react/jsx-dev-runtime.js [app-client] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$compiler$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/next/dist/compiled/react/compiler-runtime.js [app-client] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/next/dist/compiled/react/index.js [app-client] (ecmascript)");
;
var _s = __turbopack_context__.k.signature();
;
;
const ClickSpark = (t0)=>{
    _s();
    const $ = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$compiler$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["c"])(27);
    if ($[0] !== "0f646d522dc6bc826ad8147e99a42334a6de05c6414e7afce996c45013879f48") {
        for(let $i = 0; $i < 27; $i += 1){
            $[$i] = Symbol.for("react.memo_cache_sentinel");
        }
        $[0] = "0f646d522dc6bc826ad8147e99a42334a6de05c6414e7afce996c45013879f48";
    }
    const { sparkColor: t1, sparkSize: t2, sparkRadius: t3, sparkCount: t4, duration: t5, easing: t6, extraScale: t7, children } = t0;
    const sparkColor = t1 === undefined ? "#fff" : t1;
    const sparkSize = t2 === undefined ? 10 : t2;
    const sparkRadius = t3 === undefined ? 15 : t3;
    const sparkCount = t4 === undefined ? 8 : t4;
    const duration = t5 === undefined ? 400 : t5;
    const easing = t6 === undefined ? "ease-out" : t6;
    const extraScale = t7 === undefined ? 1 : t7;
    const canvasRef = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useRef"])(null);
    let t8;
    if ($[1] === Symbol.for("react.memo_cache_sentinel")) {
        t8 = [];
        $[1] = t8;
    } else {
        t8 = $[1];
    }
    const sparksRef = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useRef"])(t8);
    const startTimeRef = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useRef"])(null);
    let t10;
    let t9;
    if ($[2] === Symbol.for("react.memo_cache_sentinel")) {
        t9 = ()=>{
            const canvas = canvasRef.current;
            if (!canvas) {
                return;
            }
            const parent = canvas.parentElement || document.body;
            let resizeTimeout;
            const resizeCanvas = ()=>{
                const { width, height } = parent.getBoundingClientRect();
                if (canvas.width !== width || canvas.height !== height) {
                    canvas.width = width;
                    canvas.height = height;
                }
            };
            const handleResize = ()=>{
                clearTimeout(resizeTimeout);
                resizeTimeout = setTimeout(resizeCanvas, 100);
            };
            const ro = new ResizeObserver(handleResize);
            ro.observe(parent);
            resizeCanvas();
            return ()=>{
                ro.disconnect();
                clearTimeout(resizeTimeout);
            };
        };
        t10 = [];
        $[2] = t10;
        $[3] = t9;
    } else {
        t10 = $[2];
        t9 = $[3];
    }
    (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useEffect"])(t9, t10);
    let t11;
    if ($[4] !== easing) {
        t11 = (t)=>{
            switch(easing){
                case "linear":
                    {
                        return t;
                    }
                case "ease-in":
                    {
                        return t * t;
                    }
                case "ease-in-out":
                    {
                        return t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t;
                    }
                default:
                    {
                        return t * (2 - t);
                    }
            }
        };
        $[4] = easing;
        $[5] = t11;
    } else {
        t11 = $[5];
    }
    const easeFunc = t11;
    let t12;
    if ($[6] !== duration || $[7] !== easeFunc || $[8] !== extraScale || $[9] !== sparkColor || $[10] !== sparkRadius || $[11] !== sparkSize) {
        t12 = ()=>{
            const canvas_0 = canvasRef.current;
            if (!canvas_0) {
                return;
            }
            const ctx = canvas_0.getContext("2d");
            if (!ctx) {
                return;
            }
            let animationId;
            const draw = (timestamp)=>{
                if (!startTimeRef.current) {
                    startTimeRef.current = timestamp;
                }
                ctx.clearRect(0, 0, canvas_0.width, canvas_0.height);
                sparksRef.current = sparksRef.current.filter((spark)=>{
                    const elapsed = timestamp - spark.startTime;
                    if (elapsed >= duration) {
                        return false;
                    }
                    const progress = elapsed / duration;
                    const eased = easeFunc(progress);
                    const distance = eased * sparkRadius * extraScale;
                    const lineLength = sparkSize * (1 - eased);
                    const x1 = spark.x + distance * Math.cos(spark.angle);
                    const y1 = spark.y + distance * Math.sin(spark.angle);
                    const x2 = spark.x + (distance + lineLength) * Math.cos(spark.angle);
                    const y2 = spark.y + (distance + lineLength) * Math.sin(spark.angle);
                    ctx.strokeStyle = sparkColor;
                    ctx.lineWidth = 2;
                    ctx.beginPath();
                    ctx.moveTo(x1, y1);
                    ctx.lineTo(x2, y2);
                    ctx.stroke();
                    return true;
                });
                animationId = requestAnimationFrame(draw);
            };
            animationId = requestAnimationFrame(draw);
            return ()=>cancelAnimationFrame(animationId);
        };
        $[6] = duration;
        $[7] = easeFunc;
        $[8] = extraScale;
        $[9] = sparkColor;
        $[10] = sparkRadius;
        $[11] = sparkSize;
        $[12] = t12;
    } else {
        t12 = $[12];
    }
    let t13;
    if ($[13] !== duration || $[14] !== easeFunc || $[15] !== extraScale || $[16] !== sparkColor || $[17] !== sparkCount || $[18] !== sparkRadius || $[19] !== sparkSize) {
        t13 = [
            sparkColor,
            sparkSize,
            sparkRadius,
            sparkCount,
            duration,
            easeFunc,
            extraScale
        ];
        $[13] = duration;
        $[14] = easeFunc;
        $[15] = extraScale;
        $[16] = sparkColor;
        $[17] = sparkCount;
        $[18] = sparkRadius;
        $[19] = sparkSize;
        $[20] = t13;
    } else {
        t13 = $[20];
    }
    (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useEffect"])(t12, t13);
    let t14;
    let t15;
    if ($[21] !== sparkCount) {
        t14 = ()=>{
            const handleClick = (e)=>{
                const canvas_1 = canvasRef.current;
                if (!canvas_1) {
                    return;
                }
                const rect = canvas_1.getBoundingClientRect();
                const x = e.clientX - rect.left;
                const y = e.clientY - rect.top;
                const now = performance.now();
                const newSparks = Array.from({
                    length: sparkCount
                }, (_, i)=>({
                        x,
                        y,
                        angle: 2 * Math.PI * i / sparkCount,
                        startTime: now
                    }));
                sparksRef.current.push(...newSparks);
            };
            window.addEventListener("click", handleClick);
            return ()=>window.removeEventListener("click", handleClick);
        };
        t15 = [
            sparkCount
        ];
        $[21] = sparkCount;
        $[22] = t14;
        $[23] = t15;
    } else {
        t14 = $[22];
        t15 = $[23];
    }
    (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useEffect"])(t14, t15);
    let t16;
    if ($[24] === Symbol.for("react.memo_cache_sentinel")) {
        t16 = /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("canvas", {
            ref: canvasRef,
            className: "absolute inset-0"
        }, void 0, false, {
            fileName: "[project]/src/components/ClickSpark.tsx",
            lineNumber: 227,
            columnNumber: 11
        }, ("TURBOPACK compile-time value", void 0));
        $[24] = t16;
    } else {
        t16 = $[24];
    }
    let t17;
    if ($[25] !== children) {
        t17 = /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
            className: "pointer-events-none fixed inset-0 z-0",
            children: [
                t16,
                children
            ]
        }, void 0, true, {
            fileName: "[project]/src/components/ClickSpark.tsx",
            lineNumber: 234,
            columnNumber: 11
        }, ("TURBOPACK compile-time value", void 0));
        $[25] = children;
        $[26] = t17;
    } else {
        t17 = $[26];
    }
    return t17;
};
_s(ClickSpark, "IprSqaEqdsr6MEupqkl7p1RI3AE=");
_c = ClickSpark;
const __TURBOPACK__default__export__ = ClickSpark;
var _c;
__turbopack_context__.k.register(_c, "ClickSpark");
if (typeof globalThis.$RefreshHelpers$ === 'object' && globalThis.$RefreshHelpers !== null) {
    __turbopack_context__.k.registerExports(__turbopack_context__.m, globalThis.$RefreshHelpers$);
}
}),
"[next]/internal/font/google/doto_66c2ae84.module.css [app-client] (css module)", ((__turbopack_context__) => {

__turbopack_context__.v({
  "className": "doto_66c2ae84-module__arLeqa__className",
});
}),
"[next]/internal/font/google/doto_66c2ae84.js [app-client] (ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "default",
    ()=>__TURBOPACK__default__export__
]);
var __TURBOPACK__imported__module__$5b$next$5d2f$internal$2f$font$2f$google$2f$doto_66c2ae84$2e$module$2e$css__$5b$app$2d$client$5d$__$28$css__module$29$__ = __turbopack_context__.i("[next]/internal/font/google/doto_66c2ae84.module.css [app-client] (css module)");
;
const fontData = {
    className: __TURBOPACK__imported__module__$5b$next$5d2f$internal$2f$font$2f$google$2f$doto_66c2ae84$2e$module$2e$css__$5b$app$2d$client$5d$__$28$css__module$29$__["default"].className,
    style: {
        fontFamily: "'Doto', 'Doto Fallback'",
        fontWeight: 700,
        fontStyle: "normal"
    }
};
if (__TURBOPACK__imported__module__$5b$next$5d2f$internal$2f$font$2f$google$2f$doto_66c2ae84$2e$module$2e$css__$5b$app$2d$client$5d$__$28$css__module$29$__["default"].variable != null) {
    fontData.variable = __TURBOPACK__imported__module__$5b$next$5d2f$internal$2f$font$2f$google$2f$doto_66c2ae84$2e$module$2e$css__$5b$app$2d$client$5d$__$28$css__module$29$__["default"].variable;
}
const __TURBOPACK__default__export__ = fontData;
}),
"[project]/src/app/page.tsx [app-client] (ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "default",
    ()=>Home
]);
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/next/dist/compiled/react/jsx-dev-runtime.js [app-client] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$compiler$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/next/dist/compiled/react/compiler-runtime.js [app-client] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$src$2f$components$2f$ColorBends$2e$tsx__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/src/components/ColorBends.tsx [app-client] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$src$2f$components$2f$BlurText$2e$tsx__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/src/components/BlurText.tsx [app-client] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$src$2f$components$2f$BubbleMenu$2e$tsx__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/src/components/BubbleMenu.tsx [app-client] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$src$2f$components$2f$ClickSpark$2e$tsx__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/src/components/ClickSpark.tsx [app-client] (ecmascript)");
var __TURBOPACK__imported__module__$5b$next$5d2f$internal$2f$font$2f$google$2f$doto_66c2ae84$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[next]/internal/font/google/doto_66c2ae84.js [app-client] (ecmascript)");
"use client";
;
;
;
;
;
;
;
const items = [
    {
        label: 'Open LA²',
        href: '/la',
        ariaLabel: 'Home',
        rotation: -8,
        hoverStyles: {
            bgColor: '#3b82f6',
            textColor: '#ffffff'
        }
    },
    {
        label: 'about',
        href: '/about',
        ariaLabel: 'About',
        rotation: 8,
        hoverStyles: {
            bgColor: '#8b5cf6',
            textColor: '#ffffff'
        }
    },
    {
        label: 'contact',
        href: '/contact',
        ariaLabel: 'Contact',
        rotation: 5,
        // hoverStyles: { bgColor: '#8b5cf6', textColor: '#ffffff' }
        hoverStyles: {
            bgColor: '#10b981',
            textColor: '#ffffff'
        }
    }
];
function Home() {
    const $ = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$compiler$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["c"])(7);
    if ($[0] !== "5040d81517d769a5a4ab2714efba1d28e11c7dec0a8508be89d6d65377d3c75c") {
        for(let $i = 0; $i < 7; $i += 1){
            $[$i] = Symbol.for("react.memo_cache_sentinel");
        }
        $[0] = "5040d81517d769a5a4ab2714efba1d28e11c7dec0a8508be89d6d65377d3c75c";
    }
    const handleAnimationComplete = _HomeHandleAnimationComplete;
    let t0;
    if ($[1] === Symbol.for("react.memo_cache_sentinel")) {
        t0 = /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$src$2f$components$2f$ClickSpark$2e$tsx__$5b$app$2d$client$5d$__$28$ecmascript$29$__["default"], {
            sparkColor: "#fff",
            sparkSize: 10,
            sparkRadius: 15,
            sparkCount: 8,
            duration: 400
        }, void 0, false, {
            fileName: "[project]/src/app/page.tsx",
            lineNumber: 55,
            columnNumber: 10
        }, this);
        $[1] = t0;
    } else {
        t0 = $[1];
    }
    let t1;
    if ($[2] === Symbol.for("react.memo_cache_sentinel")) {
        t1 = /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
            className: "absolute inset-0 bg-gradient-to-b from-purple-900 via-black to-black -z-1",
            children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$src$2f$components$2f$ColorBends$2e$tsx__$5b$app$2d$client$5d$__$28$ecmascript$29$__["default"], {
                colors: [
                    "#ffffff",
                    "#ff00ff",
                    "#00ffff"
                ],
                rotation: 0,
                speed: 0.3,
                scale: 1,
                frequency: 1,
                warpStrength: 1,
                mouseInfluence: 0.1,
                parallax: 0.6,
                noise: 0.08,
                transparent: true
            }, void 0, false, {
                fileName: "[project]/src/app/page.tsx",
                lineNumber: 62,
                columnNumber: 101
            }, this)
        }, void 0, false, {
            fileName: "[project]/src/app/page.tsx",
            lineNumber: 62,
            columnNumber: 10
        }, this);
        $[2] = t1;
    } else {
        t1 = $[2];
    }
    let t2;
    let t3;
    let t4;
    if ($[3] === Symbol.for("react.memo_cache_sentinel")) {
        t2 = /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
            className: "absolute bg-black opacity-85 rounded-4xl h-full w-full blur-3xl"
        }, void 0, false, {
            fileName: "[project]/src/app/page.tsx",
            lineNumber: 71,
            columnNumber: 10
        }, this);
        t3 = /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$src$2f$components$2f$BlurText$2e$tsx__$5b$app$2d$client$5d$__$28$ecmascript$29$__["default"], {
            text: "Debenture",
            delay: 125,
            animateBy: "words",
            direction: "top",
            onAnimationComplete: handleAnimationComplete,
            className: `text-7xl md:text-8xl font-bold mb-8 text-white ${__TURBOPACK__imported__module__$5b$next$5d2f$internal$2f$font$2f$google$2f$doto_66c2ae84$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["default"].className}`
        }, void 0, false, {
            fileName: "[project]/src/app/page.tsx",
            lineNumber: 72,
            columnNumber: 10
        }, this);
        t4 = /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$src$2f$components$2f$BlurText$2e$tsx__$5b$app$2d$client$5d$__$28$ecmascript$29$__["default"], {
            text: "Intelligent Loan Sales Assistant",
            delay: 80,
            animateBy: "letters",
            direction: "top",
            onAnimationComplete: handleAnimationComplete,
            className: "text-2xl mb-8 text-white"
        }, void 0, false, {
            fileName: "[project]/src/app/page.tsx",
            lineNumber: 73,
            columnNumber: 10
        }, this);
        $[3] = t2;
        $[4] = t3;
        $[5] = t4;
    } else {
        t2 = $[3];
        t3 = $[4];
        t4 = $[5];
    }
    let t5;
    if ($[6] === Symbol.for("react.memo_cache_sentinel")) {
        t5 = /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("main", {
            className: "relative flex items-center justify-center h-screen w-full overflow-hidden",
            children: [
                t0,
                t1,
                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                    className: "relative p-3 flex flex-col items-center justify-center z-2",
                    children: [
                        t2,
                        t3,
                        t4,
                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                            className: "z-10 mt-8 w-[300px] rounded-3xl flex items-center justify-center p-4",
                            children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$src$2f$components$2f$BubbleMenu$2e$tsx__$5b$app$2d$client$5d$__$28$ecmascript$29$__["default"], {
                                items: items,
                                menuBg: "#ffffff",
                                menuContentColor: "#111111",
                                animationDuration: 0.6,
                                staggerDelay: 0.15
                            }, void 0, false, {
                                fileName: "[project]/src/app/page.tsx",
                                lineNumber: 84,
                                columnNumber: 284
                            }, this)
                        }, void 0, false, {
                            fileName: "[project]/src/app/page.tsx",
                            lineNumber: 84,
                            columnNumber: 198
                        }, this)
                    ]
                }, void 0, true, {
                    fileName: "[project]/src/app/page.tsx",
                    lineNumber: 84,
                    columnNumber: 110
                }, this)
            ]
        }, void 0, true, {
            fileName: "[project]/src/app/page.tsx",
            lineNumber: 84,
            columnNumber: 10
        }, this);
        $[6] = t5;
    } else {
        t5 = $[6];
    }
    return t5;
}
_c = Home;
function _HomeHandleAnimationComplete() {
    console.log("Animation completed!");
}
var _c;
__turbopack_context__.k.register(_c, "Home");
if (typeof globalThis.$RefreshHelpers$ === 'object' && globalThis.$RefreshHelpers !== null) {
    __turbopack_context__.k.registerExports(__turbopack_context__.m, globalThis.$RefreshHelpers$);
}
}),
]);

//# sourceMappingURL=%5Broot-of-the-server%5D__5a8b68d7._.js.map