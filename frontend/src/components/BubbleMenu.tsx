"use client";
import type { CSSProperties } from "react";
import { useLayoutEffect, useRef } from "react";
import { gsap } from "gsap";

type MenuItem = {
  label: string;
  href: string;
  ariaLabel?: string;
  rotation?: number;
  hoverStyles?: { bgColor?: string; textColor?: string };
};

export type BubbleMenuProps = {
  items?: MenuItem[];
  className?: string;
  style?: CSSProperties;
  menuBg?: string;
  menuContentColor?: string;
  animationDuration?: number;
  staggerDelay?: number; // keep if you still want sequential appearance
};

export default function BubbleMenu({
  items = [],
  className,
  style,
  menuBg = "#fff",
  menuContentColor = "#111",
  animationDuration = 0.6,
  staggerDelay = 0.15,
}: BubbleMenuProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);

  useLayoutEffect(() => {
    const ctx = gsap.context(() => {
      const container = containerRef.current;
      if (!container) return;

      const bubbles = Array.from(
        container.querySelectorAll<HTMLAnchorElement>(".bubble-item")
      );
      if (!bubbles.length) return;

      // 1) stop any previous tweens & inline transforms (refresh/HMR safe)
      gsap.killTweensOf(bubbles);
      gsap.set(bubbles, { clearProps: "transform" });

      // 2) ALL start at scale 0 at FRAME 0 (before paint), GSAP owns rotate
      gsap.set(bubbles, {
        scale: 0,
        transformOrigin: "50% 50%",
        rotate: (_, el) => Number(el.getAttribute("data-rotate")) || 0,
        force3D: true,
      });

      // 3) Animate ONLY scale → 1 (no alpha/opacity)
      gsap.to(bubbles, {
        scale: 1,
        duration: animationDuration,
        ease: "back.out(1.7)", // nice pop, no overshoot flicker
        // If you want ALL finish together, remove stagger:
        // stagger: 0
        // If you want one-by-one, keep stagger:
        stagger: staggerDelay,
      });
    }, containerRef);

    return () => ctx.revert();
  }, [items, animationDuration, staggerDelay]);

  return (
    <div
      ref={containerRef}
      className={[
        "bubble-inline-menu flex justify-center items-center flex-wrap gap-3",
        className,
      ].filter(Boolean).join(" ")}
      style={style}
    >
      {items.map((item, idx) => (
        <a
          key={idx}
          href={item.href}
          aria-label={item.ariaLabel || item.label}
          className={
            // IMPORTANT: do NOT use transition-all (it would animate transform too)
            "bubble-item rounded-full px-6 py-3 text-lg md:text-xl font-semibold " +
            "transition-[background,color,box-shadow] duration-300 " + // only these
            "shadow-md hover:shadow-lg will-change-transform"
          }
          data-rotate={item.rotation ?? 0}
          style={
            {
              background: menuBg,
              color: menuContentColor,
            } as CSSProperties
          }
          onMouseEnter={(e) => {
            const el = e.currentTarget as HTMLElement;
            el.style.background = item.hoverStyles?.bgColor || "#ddd";
            el.style.color = item.hoverStyles?.textColor || menuContentColor;
          }}
          onMouseLeave={(e) => {
            const el = e.currentTarget as HTMLElement;
            el.style.background = menuBg;
            el.style.color = menuContentColor;
          }}
        >
          <span className="bubble-label inline-block">{item.label}</span>
        </a>
      ))}
    </div>
  );
}
