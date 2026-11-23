import React, { useEffect, useRef } from "react";
import { animate, stagger } from "animejs";

export interface BlurTextProps {
  text: string;
  className?: string;
  as?: "p" | "h2" | "h3" | "span";
  delayMs?: number;
}

/**
 * React Bits style BlurText word reveal using Anime.js v4.
 */
export const BlurText: React.FC<BlurTextProps> = ({
  text,
  className = "",
  as = "p",
  delayMs = 140,
}) => {
  const ref = useRef<HTMLElement | null>(null);
  const words = text.split(" ");

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const spans = el.querySelectorAll<HTMLElement>("[data-blur-word]");
    if (reduced || spans.length === 0) return;

    const anim = animate(spans, {
      opacity: [0, 1],
      filter: ["blur(6px)", "blur(0px)"],
      y: [6, 0],
      duration: 480,
      delay: stagger(32, { start: delayMs }),
      ease: "outQuad",
    });

    return () => {
      anim.cancel();
    };
  }, [text, delayMs]);

  const Tag = as as React.ElementType;
  return (
    <Tag ref={ref} className={className}>
      {words.map((w, idx) => (
        <span key={`${w}-${idx}`} data-blur-word className="inline-block mr-[0.32em] last:mr-0">
          {w}
        </span>
      ))}
    </Tag>
  );
};
