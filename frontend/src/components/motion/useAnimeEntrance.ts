import { useEffect, type RefObject } from "react";
import { createTimeline, stagger } from "animejs";

/**
 * Coordinated Anime.js v4 stagger entrance timeline hook with reduced-motion & cleanup.
 */
export function useAnimeStagger(
  containerRef: RefObject<HTMLElement | null>,
  selector: string,
  triggerKey: unknown
): void {
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const items = el.querySelectorAll<HTMLElement>(selector);
    if (reduced || items.length === 0) return;

    const tl = createTimeline({
      defaults: { duration: 420, ease: "outCubic" },
    });
    tl.add(items, {
      opacity: [0, 1],
      y: [10, 0],
      delay: stagger(42),
    });

    return () => {
      tl.cancel();
    };
  }, [containerRef, selector, triggerKey]);
}
