import React from "react";

export interface TocHeading {
  level: number;
  text: string;
  id: string;
}

function slugify(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s-]/g, "")
    .trim()
    .replace(/\s+/g, "-");
}

export function extractHeadings(markdown: string): TocHeading[] {
  const headings: TocHeading[] = [];
  const lines = markdown.replace(/\r\n/g, "\n").split("\n");
  let inCode = false;
  for (const line of lines) {
    if (line.trim().startsWith("```")) {
      inCode = !inCode;
      continue;
    }
    if (inCode) continue;
    const m = /^(#{1,6})\s+(.+)$/.exec(line.trim());
    if (m) {
      const cleanText = m[2].replace(/[*_`]/g, "").trim();
      headings.push({
        level: m[1].length,
        text: cleanText,
        id: slugify(cleanText),
      });
    }
  }
  return headings;
}

/**
 * Recursive single-pass inline parser supporting `code`, **bold**, *italic*, and [links](url)
 * even when code spans appear inside bold or link text.
 */
function renderInline(text: string, depth = 0): React.ReactNode[] {
  if (depth > 6 || !text) return [text];

  const out: React.ReactNode[] = [];
  let i = 0;
  let k = 0;

  const flushPlain = (chunk: string) => {
    if (chunk) out.push(chunk);
  };

  let buf = "";
  while (i < text.length) {
    // 1. Inline code span: `...`
    if (text[i] === "`") {
      const end = text.indexOf("`", i + 1);
      if (end !== -1) {
        flushPlain(buf);
        buf = "";
        const codeContent = text.slice(i + 1, end);
        out.push(
          <code
            key={k++}
            className="px-1.5 py-0.5 mx-0.5 bg-[#202321] border border-[#303431] rounded-[2px] font-mono-tech text-[11.5px] text-[#EAE9E4]"
          >
            {codeContent}
          </code>
        );
        i = end + 1;
        continue;
      }
    }

    // 2. Bold: **...**
    if (text[i] === "*" && text[i + 1] === "*") {
      const end = text.indexOf("**", i + 2);
      if (end !== -1) {
        flushPlain(buf);
        buf = "";
        const inner = text.slice(i + 2, end);
        out.push(
          <strong key={k++} className="font-semibold text-[#EAE9E4]">
            {renderInline(inner, depth + 1)}
          </strong>
        );
        i = end + 2;
        continue;
      }
    }

    // 3. Single italic: *...* (must not start with **)
    if (text[i] === "*" && text[i + 1] !== "*") {
      let searchFrom = i + 1;
      let end = -1;
      while (searchFrom < text.length) {
        const candidate = text.indexOf("*", searchFrom);
        if (candidate === -1) break;
        if (text[candidate + 1] === "*") {
          // Skip past closing ** owned by outer bold span or nested bold
          searchFrom = candidate + 2;
          continue;
        }
        end = candidate;
        break;
      }
      if (end !== -1 && text[i + 1] !== " " && text[end - 1] !== " ") {
        flushPlain(buf);
        buf = "";
        const inner = text.slice(i + 1, end);
        out.push(
          <em key={k++} className="italic text-[#EAE9E4]/90">
            {renderInline(inner, depth + 1)}
          </em>
        );
        i = end + 1;
        continue;
      }
    }

    // 4. Link: [label](url)
    if (text[i] === "[") {
      const closeBracket = text.indexOf("]", i + 1);
      if (closeBracket !== -1 && text[closeBracket + 1] === "(") {
        const closeParen = text.indexOf(")", closeBracket + 2);
        if (closeParen !== -1) {
          flushPlain(buf);
          buf = "";
          const label = text.slice(i + 1, closeBracket);
          const href = text.slice(closeBracket + 2, closeParen).trim();
          const isHttp = href.startsWith("http://") || href.startsWith("https://");
          if (isHttp) {
            out.push(
              <a
                key={k++}
                href={href}
                target="_blank"
                rel="noreferrer"
                className="text-[#C87951] hover:text-[#EAE9E4] underline underline-offset-2"
              >
                {renderInline(label, depth + 1)}
              </a>
            );
          } else {
            out.push(
              <code
                key={k++}
                className="px-1.5 py-0.5 mx-0.5 bg-[#202321] border border-[#303431] rounded-[2px] font-mono-tech text-[11.5px] text-[#C87951]"
                title={href}
              >
                {label}
              </code>
            );
          }
          i = closeParen + 1;
          continue;
        }
      }
    }

    buf += text[i];
    i++;
  }

  flushPlain(buf);
  return out;
}

function isTableDivider(line: string): boolean {
  const s = line.trim();
  return s.includes("|") && /^\|?[\s:|-]+\|?$/.test(s) && s.includes("---");
}

function splitTableRow(line: string): string[] {
  const trimmed = line.trim().replace(/^\|/, "").replace(/\|$/, "");
  return trimmed.split("|").map((cell) => cell.trim());
}

export const MarkdownRenderer: React.FC<{ content: string }> = ({ content }) => {
  const lines = content.replace(/\r\n/g, "\n").split("\n");
  const blocks: React.ReactNode[] = [];
  let i = 0;
  let blockKey = 0;

  while (i < lines.length) {
    const rawLine = lines[i];
    const line = rawLine.trim();

    if (!line) {
      i++;
      continue;
    }

    // Fenced code blocks
    if (line.startsWith("```")) {
      const lang = line.slice(3).trim();
      const codeLines: string[] = [];
      i++;
      while (i < lines.length && !lines[i].trim().startsWith("```")) {
        codeLines.push(lines[i]);
        i++;
      }
      i++;
      blocks.push(
        <div key={blockKey++} className="my-3 border border-[#303431] bg-[#0D0E0E] rounded-[2px] overflow-hidden">
          {lang && (
            <div className="px-3 py-1 border-b border-[#303431] bg-[#121414] text-[10px] font-mono-tech uppercase text-[#696E68]">
              {lang}
            </div>
          )}
          <pre className="p-3.5 overflow-x-auto text-[11.5px] font-mono-tech text-[#EAE9E4] leading-relaxed">
            <code>{codeLines.join("\n")}</code>
          </pre>
        </div>
      );
      continue;
    }

    // Display math block $$...$$
    if (line.startsWith("$$")) {
      const mathLines: string[] = [line.replace(/^\$\$/, "")];
      if (!line.endsWith("$$") || line === "$$") {
        i++;
        while (i < lines.length && !lines[i].trim().endsWith("$$")) {
          mathLines.push(lines[i]);
          i++;
        }
        if (i < lines.length) {
          mathLines.push(lines[i].trim().replace(/\$\$$/, ""));
          i++;
        }
      } else {
        mathLines[0] = mathLines[0].replace(/\$\$$/, "");
        i++;
      }
      blocks.push(
        <pre
          key={blockKey++}
          className="my-3 px-3.5 py-2.5 bg-[#121414] border border-[#303431] rounded-[2px] font-mono-tech text-[12px] text-[#EAE9E4] overflow-x-auto"
        >
          {mathLines.join(" ").trim()}
        </pre>
      );
      continue;
    }

    // Horizontal rule
    if (/^(-{3,}|\*{3,}|_{3,})$/.test(line)) {
      blocks.push(<hr key={blockKey++} className="my-5 border-[#303431]" />);
      i++;
      continue;
    }

    // Headings (# .. ######)
    const headingMatch = /^(#{1,6})\s+(.+)$/.exec(line);
    if (headingMatch) {
      const level = headingMatch[1].length;
      const cleanTitle = headingMatch[2].replace(/[*_`]/g, "").trim();
      const id = slugify(cleanTitle);
      if (level === 1) {
        blocks.push(
          <h2 key={blockKey++} id={id} className="text-[20px] font-semibold text-[#EAE9E4] mt-2 mb-3 pb-2 border-b border-[#303431]">
            {renderInline(headingMatch[2])}
          </h2>
        );
      } else if (level === 2) {
        blocks.push(
          <h3 key={blockKey++} id={id} className="text-[16px] font-semibold text-[#EAE9E4] mt-6 mb-2.5 pb-1.5 border-b border-[#303431]/70">
            {renderInline(headingMatch[2])}
          </h3>
        );
      } else if (level === 3) {
        blocks.push(
          <h4 key={blockKey++} id={id} className="text-[14px] font-semibold text-[#C87951] mt-4 mb-2">
            {renderInline(headingMatch[2])}
          </h4>
        );
      } else {
        blocks.push(
          <h5 key={blockKey++} id={id} className="text-[13px] font-semibold text-[#EAE9E4] mt-3 mb-1.5">
            {renderInline(headingMatch[2])}
          </h5>
        );
      }
      i++;
      continue;
    }

    // Blockquote
    if (line.startsWith(">")) {
      const quoteLines: string[] = [];
      while (i < lines.length && lines[i].trim().startsWith(">")) {
        quoteLines.push(lines[i].trim().replace(/^>\s?/, ""));
        i++;
      }
      blocks.push(
        <blockquote
          key={blockKey++}
          className="my-3 pl-3.5 py-2 pr-3 border-l-2 border-[#C87951] bg-[#121414] text-[12.5px] text-[#9B9E98] leading-relaxed"
        >
          {renderInline(quoteLines.join(" "))}
        </blockquote>
      );
      continue;
    }

    // Markdown table
    if (line.includes("|") && i + 1 < lines.length && isTableDivider(lines[i + 1])) {
      const headers = splitTableRow(rawLine);
      i += 2;
      const rows: string[][] = [];
      while (i < lines.length && lines[i].trim().includes("|") && lines[i].trim() !== "") {
        rows.push(splitTableRow(lines[i]));
        i++;
      }
      blocks.push(
        <div key={blockKey++} className="my-3 table-wrapper border border-[#303431] rounded-[2px]" tabIndex={0} role="region">
          <table className="w-full text-left text-[12px] border-collapse">
            <thead>
              <tr className="border-b border-[#303431] bg-[#121414] text-[10.5px] font-mono-tech uppercase text-[#9B9E98]">
                {headers.map((h, hi) => (
                  <th key={hi} className="py-2 px-3">
                    {renderInline(h)}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-[#303431]/70 text-[#EAE9E4]/90">
              {rows.map((row, ri) => (
                <tr key={ri} className="hover:bg-[#202321]/50">
                  {headers.map((_, ci) => (
                    <td key={ci} className="py-2 px-3 align-top">
                      {renderInline(row[ci] ?? "")}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      );
      continue;
    }

    // Unordered list (supports nested/indented sub-bullets)
    if (/^\s*[-*+]\s+/.test(rawLine)) {
      const items: Array<{ indent: number; text: string }> = [];
      while (i < lines.length && /^\s*[-*+]\s+/.test(lines[i])) {
        const lm = /^(\s*)[-*+]\s+(.+)$/.exec(lines[i]);
        if (lm) {
          items.push({ indent: lm[1].length, text: lm[2].trim() });
        }
        i++;
      }
      blocks.push(
        <ul key={blockKey++} className="my-2.5 space-y-1.5 list-disc pl-5 text-[13px] text-[#9B9E98] leading-relaxed">
          {items.map((item, idx) => (
            <li key={idx} className={item.indent >= 2 ? "ml-4 text-[12.5px]" : ""}>
              {renderInline(item.text)}
            </li>
          ))}
        </ul>
      );
      continue;
    }

    // Ordered list
    if (/^\s*\d+\.\s+/.test(rawLine)) {
      const items: string[] = [];
      while (i < lines.length && /^\s*\d+\.\s+/.test(lines[i])) {
        items.push(lines[i].trim().replace(/^\d+\.\s+/, ""));
        i++;
      }
      blocks.push(
        <ol key={blockKey++} className="my-2.5 space-y-1.5 list-decimal pl-5 text-[13px] text-[#9B9E98] leading-relaxed">
          {items.map((item, idx) => (
            <li key={idx}>{renderInline(item)}</li>
          ))}
        </ol>
      );
      continue;
    }

    // Paragraph (join contiguous non-structural lines)
    const paraLines: string[] = [line];
    i++;
    while (
      i < lines.length &&
      lines[i].trim() !== "" &&
      !/^(#{1,6}\s+|```|\$\$|>|\s*[-*+]\s+|\s*\d+\.\s+|-{3,}$)/.test(lines[i].trim()) &&
      !(lines[i].includes("|") && i + 1 < lines.length && isTableDivider(lines[i + 1]))
    ) {
      paraLines.push(lines[i].trim());
      i++;
    }
    blocks.push(
      <p key={blockKey++} className="my-2.5 text-[13.5px] text-[#9B9E98] leading-relaxed">
        {renderInline(paraLines.join(" "))}
      </p>
    );
  }

  return <div className="max-w-[78ch]">{blocks}</div>;
};
