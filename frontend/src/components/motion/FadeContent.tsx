import React, { useEffect, useRef } from "react";
import { animate } from "animejs";

export interface FadeContentProps {
  children: React.ReactNode;
  className?: string;
  delayMs?: number;
  yOffset?: number;
}

/**
 * Restrained scroll/mount content reveal matching planetary mission UI restraint.
 */
export const FadeContent: React.FC<FadeContentProps> = ({
  children,
  className = "",
  delayMs = 0,
  yOffset = 10,
}) => {
  const ref = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (reduced) return;

    const anim = animate(el, {
      opacity: [0, 1],
      y: [yOffset, 0],
      duration: 420,
      delay: delayMs,
      ease: "outCubic",
    });

    return () => {
      anim.cancel();
    };
  }, [delayMs, yOffset]);

  return (
    <div ref={ref} className={className}>
      {children}
    </div>
  );
};
