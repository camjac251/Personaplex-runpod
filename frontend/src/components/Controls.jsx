import { useEffect, useId, useRef, useState } from "react";
import { createPortal } from "react-dom";

import { PARAM_INFO } from "../data/dashboardData.jsx";
import { cls } from "../utils/format.js";

export function Info({ k }) {
  const meta = PARAM_INFO[k];
  const ref = useRef(null);
  const timer = useRef(null);
  const [open, setOpen] = useState(false);
  const [pos, setPos] = useState({ top: 0, left: 0, flip: false });

  const closeSoon = () => {
    clearTimeout(timer.current);
    timer.current = setTimeout(() => setOpen(false), 120);
  };

  const keepOpen = () => {
    clearTimeout(timer.current);
    setOpen(true);
  };

  useEffect(() => {
    if (!open) return;
    const update = () => {
      const rect = ref.current?.getBoundingClientRect();
      if (!rect) return;
      const width = 250;
      const margin = 10;
      let left = rect.left + rect.width / 2 - width / 2;
      left = Math.max(margin, Math.min(window.innerWidth - width - margin, left));
      const flip = rect.top < 150;
      setPos({ left, top: flip ? rect.bottom + 8 : rect.top - 8, flip });
    };
    update();
    window.addEventListener("resize", update);
    window.addEventListener("scroll", update, true);
    return () => {
      window.removeEventListener("resize", update);
      window.removeEventListener("scroll", update, true);
    };
  }, [open]);

  if (!meta) return null;
  const tipPanel = (
    <div
      className={cls("info-tip-portal", pos.flip && "below")}
      role="tooltip"
      style={{
        position: "fixed",
        left: pos.left,
        top: pos.top,
        transform: pos.flip ? "translateY(0)" : "translateY(-100%)",
      }}
      onMouseEnter={keepOpen}
      onMouseLeave={closeSoon}
    >
      <div className="info-tip-title">{meta.title}</div>
      <div className="info-tip-body">{meta.body}</div>
    </div>
  );
  return (
    <>
      <button
        ref={ref}
        type="button"
        className={cls("tipbtn", open && "active")}
        aria-label={`Info: ${meta.title}`}
        onMouseEnter={keepOpen}
        onMouseLeave={closeSoon}
        onClick={(event) => {
          event.preventDefault();
          event.stopPropagation();
          setOpen((current) => !current);
        }}
      />
      {open && createPortal(tipPanel, document.body)}
    </>
  );
}

export function Listbox({ value, options, onChange, placeholder = "Select", label = placeholder, caption, info, disabled = false }) {
  const ref = useRef(null);
  const menuId = useId();
  const [open, setOpen] = useState(false);
  const [focusIdx, setFocusIdx] = useState(0);
  const current = options.find((option) => option.value === value);
  const activeIndex = options.findIndex((option) => option.value === value);

  useEffect(() => {
    if (!open) return;
    setFocusIdx(Math.max(0, activeIndex));
    const onDoc = (event) => {
      if (!ref.current?.contains(event.target)) setOpen(false);
    };
    document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, [activeIndex, open]);

  const choose = (option) => {
    if (!option || disabled) return;
    onChange(option.value);
    setOpen(false);
  };

  const onKeyDown = (event) => {
    if (!["ArrowDown", "ArrowUp", "Home", "End", "Enter", " ", "Escape"].includes(event.key)) return;
    if (!open && ["ArrowDown", "ArrowUp", "Enter", " "].includes(event.key)) {
      event.preventDefault();
      setOpen(true);
      setFocusIdx(Math.max(0, activeIndex));
      return;
    }
    if (!open) return;
    if (event.key === "Escape") {
      event.preventDefault();
      setOpen(false);
      return;
    }
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      choose(options[focusIdx]);
      return;
    }
    event.preventDefault();
    if (event.key === "Home") setFocusIdx(0);
    if (event.key === "End") setFocusIdx(Math.max(0, options.length - 1));
    if (event.key === "ArrowDown") setFocusIdx((index) => Math.min(options.length - 1, index + 1));
    if (event.key === "ArrowUp") setFocusIdx((index) => Math.max(0, index - 1));
  };

  const field = (
    <div className="lb" ref={ref}>
      <button
        type="button"
        role="combobox"
        className={cls("lb-trigger", open && "open")}
        disabled={disabled}
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-controls={menuId}
        aria-activedescendant={open ? `${menuId}-opt-${focusIdx}` : undefined}
        aria-label={label}
        onKeyDown={onKeyDown}
        onClick={() => {
          if (!disabled) setOpen((currentOpen) => !currentOpen);
        }}
      >
        <span className="lb-trigger-label">{current?.label || placeholder}</span>
        <svg className="lb-chev" viewBox="0 0 10 10" aria-hidden="true" focusable="false">
          <path
            d="M2 4l3 3 3-3"
            stroke="currentColor"
            fill="none"
            strokeWidth="1.4"
            strokeLinecap="round"
          />
        </svg>
      </button>
      {open && (
        <div id={menuId} className="lb-menu" role="listbox" aria-label={label}>
          {options.map((option, index) => (
            <button
              key={option.value}
              id={`${menuId}-opt-${index}`}
              type="button"
              className={cls("lb-opt", option.value === value && "active", index === focusIdx && "focused")}
              role="option"
              aria-selected={option.value === value}
              onMouseEnter={() => setFocusIdx(index)}
              onClick={() => choose(option)}
            >
              <span className="lb-opt-marker" />
              <span className="lb-opt-body">
                <span className="lb-opt-label">{option.label}</span>
                {option.desc && <span className="lb-opt-desc">{option.desc}</span>}
              </span>
              {option.tag && <span className="lb-opt-tag">{option.tag}</span>}
            </button>
          ))}
        </div>
      )}
    </div>
  );
  if (!caption && !info) return field;
  return (
    <div className="lb-field">
      <div className="lb-caption">
        <span>{caption || label}</span>
        {info && <Info k={info} />}
      </div>
      {field}
    </div>
  );
}

export function ToggleRow({ name, desc, value, onChange, info }) {
  return (
    <div className="toggle">
      <div className="l">
        <div className="n" style={{ display: "inline-flex", alignItems: "center" }}>
          {name}
          {info && <Info k={info} />}
        </div>
        {desc && <div className="d">{desc}</div>}
      </div>
      <button
        type="button"
        className={cls("switch", value && "on")}
        role="switch"
        aria-checked={value}
        aria-label={name}
        onClick={() => onChange(!value)}
      />
    </div>
  );
}

export function MiniSlider({ label, value, onChange, onCommit, min, max, step, format = (v) => v, info }) {
  // Value when the current pointer/keyboard interaction began. A native
  // range input teleports to the click position (and to min/max on
  // Home/End), so commit handlers get the start value as a second
  // argument to detect a single interaction jumping past a safety band
  // and to revert it cleanly.
  const interactionStartRef = useRef(null);
  const begin = () => {
    if (interactionStartRef.current == null) {
      interactionStartRef.current = Number(value);
    }
  };
  const commit = (event) => {
    const startValue = interactionStartRef.current;
    interactionStartRef.current = null;
    onCommit?.(Number(event.currentTarget.value), startValue);
  };
  return (
    <div className="mini">
      <div className="mini-row">
        <span className="l" style={{ display: "inline-flex", alignItems: "center" }}>
          {label}
          {info && <Info k={info} />}
        </span>
        <span className="v">{format(value)}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        aria-label={label}
        onPointerDown={begin}
        onKeyDown={begin}
        onChange={(event) => onChange(Number(event.target.value))}
        onPointerUp={commit}
        onKeyUp={commit}
        onBlur={commit}
      />
    </div>
  );
}
