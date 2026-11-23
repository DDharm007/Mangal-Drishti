import React, { useEffect, useRef } from "react";
import { animate, stagger } from "animejs";

export interface SplitTextProps {
  text: string;
  className?: string;
  as?: "h1" | "h2" | "h3" | "span";
  delayMs?: number;
  staggerMs?: number;
}

/**
 * React Bits style SplitText entrance powered by Anime.js v4.
 * Respects prefers-reduced-motion and cleans up instances on unmount.
 */
export const SplitText: React.FC<SplitTextProps> = ({
  text,
  className = "",
  as = "h1",
  delayMs = 40,
  staggerMs = 28,
}) => {
  const containerRef = useRef<HTMLElement | null>(null);
  const words = text.split(" ");

  useEffect(() => {
    const root = containerRef.current;
    if (!root) return;
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const chars = root.querySelectorAll<HTMLElement>("[data-split-char]");
    if (reduced || chars.length === 0) return;

    const anim = animate(chars, {
      opacity: [0, 1],
      y: [12, 0],
      duration: 520,
      delay: stagger(staggerMs, { start: delayMs }),
      ease: "outCubic",
    });

    return () => {
      anim.cancel();
    };
  }, [text, delayMs, staggerMs]);

  const Tag = as as React.ElementType;
  return (
    <Tag ref={containerRef} className={className} aria-label={text}>
      {words.map((word, wIdx) => (
        <span key={`${word}-${wIdx}`} className="inline-block whitespace-nowrap mr-[0.28em] last:mr-0">
          {Array.from(word).map((ch, cIdx) => (
            <span
              key={`${ch}-${cIdx}`}
              data-split-char
              className="inline-block will-change-transform"
              aria-hidden="true"
            >
              {ch}
            </span>
          ))}
        </span>
      ))}
    </Tag>
  );
};
