import { useId, type SVGProps } from "react";

interface LogoProps extends SVGProps<SVGSVGElement> {
  size?: number | string;
}

export default function Logo({
  size = 32,
  className = "",
  style,
  ...props
}: LogoProps) {
  const rawId = useId();
  const maskId = `logo-mask-${rawId.replace(/:/g, "")}`;

  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 205 200"
      width={size}
      height={size}
      className={`shrink-0 ${className}`}
      style={{
        color: "currentColor",
        ...style,
      }}
      {...props}
    >
      <defs>
        <mask id={maskId}>
          <rect width="205" height="200" fill="white" />
          <path
            d="M 100 72 C 82 72, 74 80, 80 90 C 68 92, 68 108, 80 110 C 74 120, 82 128, 100 128 C 118 128, 126 120, 120 110 C 132 108, 132 92, 120 90 C 126 80, 118 72, 100 72 Z"
            fill="black"
            stroke="black"
            strokeWidth="8"
            strokeLinejoin="round"
          />
          <path
            d="M 50 182 L 145 182 L 155 172"
            fill="none"
            stroke="black"
            strokeWidth="4"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
          <line
            x1="100"
            y1="65"
            x2="100"
            y2="80"
            stroke="black"
            strokeWidth="5"
          />
          <line
            x1="100"
            y1="120"
            x2="100"
            y2="135"
            stroke="black"
            strokeWidth="5"
          />
          <rect x="97.5" y="97.5" width="5" height="5" fill="black" />
        </mask>
      </defs>

      <g mask={`url(#${maskId})`}>
        <path
          d="M 55 10 L 80 10 C 83 10, 85 12, 85 15 L 85 170 L 155 170 L 175 190 L 35 190 L 55 170 Z"
          fill="currentColor"
        />
        <path
          d="M 115 15 C 115 12, 117 10, 120 10 L 145 10 L 145 150 L 115 150 Z"
          fill="currentColor"
        />
        <rect x="20" y="15" width="30" height="10" fill="currentColor" />
        <rect x="5" y="30" width="20" height="10" fill="currentColor" />
        <rect x="30" y="30" width="20" height="10" fill="currentColor" />
        <rect x="15" y="45" width="15" height="10" fill="currentColor" />
        <rect x="35" y="45" width="15" height="10" fill="currentColor" />
        <rect x="150" y="15" width="30" height="10" fill="currentColor" />
        <rect x="185" y="15" width="15" height="10" fill="currentColor" />
        <rect x="150" y="30" width="20" height="10" fill="currentColor" />
        <rect x="175" y="30" width="25" height="10" fill="currentColor" />
        <rect x="150" y="45" width="15" height="10" fill="currentColor" />
        <rect x="170" y="45" width="20" height="10" fill="currentColor" />
      </g>

      <path
        d="M 100 72 C 82 72, 74 80, 80 90 C 68 92, 68 108, 80 110 C 74 120, 82 128, 100 128 C 118 128, 126 120, 120 110 C 132 108, 132 92, 120 90 C 126 80, 118 72, 100 72 Z"
        fill="none"
        stroke="currentColor"
        strokeWidth="3.5"
        strokeLinejoin="round"
      />
      <rect x="94" y="94" width="12" height="12" fill="currentColor" />
      <g
        fill="none"
        stroke="currentColor"
        strokeWidth="3"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <path d="M 85 84 L 97 84 L 97 94" />
        <path d="M 77 100 L 94 100" />
        <path d="M 85 116 L 97 116 L 97 106" />
        <path d="M 115 84 L 103 84 L 103 94" />
        <path d="M 123 100 L 106 100" />
        <path d="M 115 116 L 103 116 L 103 106" />
      </g>
      <g fill="currentColor">
        <circle cx="85" cy="84" r="3.2" />
        <circle cx="77" cy="100" r="3.2" />
        <circle cx="85" cy="116" r="3.2" />
        <circle cx="115" cy="84" r="3.2" />
        <circle cx="123" cy="100" r="3.2" />
        <circle cx="115" cy="116" r="3.2" />
      </g>
    </svg>
  );
}
