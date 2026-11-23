import React, { useRef } from "react";

export interface SpotlightSurfaceProps extends React.HTMLAttributes<HTMLDivElement> {
  children: React.ReactNode;
  className?: string;
}

/**
 * React Bits style restrained surface hover illumination (zero React state re-renders).
 */
export const SpotlightSurface: React.FC<SpotlightSurfaceProps> = ({
  children,
  className = "",
  onPointerMove,
  onPointerLeave,
  ...rest
}) => {
  const ref = useRef<HTMLDivElement | null>(null);

  const handleMove = (e: React.PointerEvent<HTMLDivElement>) => {
    const el = ref.current;
    if (el) {
      const rect = el.getBoundingClientRect();
      el.style.setProperty("--spot-x", `${e.clientX - rect.left}px`);
      el.style.setProperty("--spot-y", `${e.clientY - rect.top}px`);
      el.style.setProperty("--spot-opacity", "1");
    }
    onPointerMove?.(e);
  };

  const handleLeave = (e: React.PointerEvent<HTMLDivElement>) => {
    const el = ref.current;
    if (el) {
      el.style.setProperty("--spot-opacity", "0");
    }
    onPointerLeave?.(e);
  };

  return (
    <div
      ref={ref}
      onPointerMove={handleMove}
      onPointerLeave={handleLeave}
      className={`spotlight-surface ${className}`}
      {...rest}
    >
      {children}
    </div>
  );
};
