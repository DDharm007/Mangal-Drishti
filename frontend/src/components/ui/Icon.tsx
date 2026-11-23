import React from "react";

export interface IconProps {
  name: string;
  variant?: "solid" | "regular";
  className?: string;
  spin?: boolean;
  title?: string;
}

/**
 * Unified Font Awesome Kit icon wrapper.
 * Loads once from root index.html without mixing competing icon libraries or emojis.
 */
export const Icon: React.FC<IconProps> = ({
  name,
  variant = "solid",
  className = "",
  spin = false,
  title,
}) => {
  const prefix = variant === "regular" ? "fa-regular" : "fa-solid";
  const spinClass = spin ? "fa-spin" : "";
  return (
    <i
      className={`${prefix} fa-${name} ${spinClass} ${className}`.trim()}
      aria-hidden="true"
      title={title}
    />
  );
};
