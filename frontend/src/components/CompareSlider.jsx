import { useRef, useState } from "react";

/** Before/after slider. `left` shows on the left of the divider, `right` on the right.
 *  The range input makes it keyboard- and screen-reader-accessible. */
export default function CompareSlider({ left, right, leftLabel, rightLabel, aspect }) {
  const [pos, setPos] = useState(50);
  const box = useRef(null);
  const dragging = useRef(false);

  const move = (clientX) => {
    const r = box.current.getBoundingClientRect();
    setPos(Math.min(100, Math.max(0, ((clientX - r.left) / r.width) * 100)));
  };

  return (
    <div
      className="slider"
      ref={box}
      style={{ aspectRatio: aspect }}
      onPointerDown={(e) => {
        dragging.current = true;
        e.currentTarget.setPointerCapture(e.pointerId);
        move(e.clientX);
      }}
      onPointerMove={(e) => dragging.current && move(e.clientX)}
      onPointerUp={() => (dragging.current = false)}
    >
      <img src={right} alt={rightLabel} draggable={false} />
      <img src={left} alt={leftLabel} draggable={false} style={{ clipPath: `inset(0 ${100 - pos}% 0 0)` }} />
      <span className="tag tag-l">{leftLabel}</span>
      <span className="tag tag-r">{rightLabel}</span>
      <div className="divider" style={{ left: `${pos}%` }}>
        <span className="knob" aria-hidden="true">⇆</span>
      </div>
      <input
        type="range" min="0" max="100" step="1" value={Math.round(pos)}
        onChange={(e) => setPos(Number(e.target.value))}
        aria-label={`Reveal ${rightLabel} over ${leftLabel}`}
        className="slider-input"
      />
    </div>
  );
}
