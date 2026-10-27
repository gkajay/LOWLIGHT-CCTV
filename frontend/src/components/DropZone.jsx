import { useId, useState } from "react";

export default function DropZone({ file, onFile, label, hint, compact = false }) {
  const id = useId();
  const [over, setOver] = useState(false);
  const accept = (f) => {
    if (f && f.type.startsWith("image/")) onFile(f);
  };
  return (
    <label
      htmlFor={id}
      className={`drop${over ? " over" : ""}${compact ? " compact" : ""}`}
      onDragOver={(e) => { e.preventDefault(); setOver(true); }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => { e.preventDefault(); setOver(false); accept(e.dataTransfer.files[0]); }}
    >
      <input id={id} type="file" accept="image/png,image/jpeg,image/bmp" onChange={(e) => accept(e.target.files[0])} />
      {file ? (
        <>
          <strong>{file.name}</strong>
          <span>{(file.size / 1024).toFixed(0)} KB · click or drop to replace</span>
        </>
      ) : (
        <>
          <strong>{label}</strong>
          <span>{hint}</span>
        </>
      )}
    </label>
  );
}
