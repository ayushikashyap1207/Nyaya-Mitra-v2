import os
import gradio as gr
import tempfile


def safe_text(text: str) -> str:
    return (
        str(text)
        .replace("“", '"')
        .replace("”", '"')
        .replace("’", "'")
        .encode("utf-8", "ignore")
        .decode("utf-8")
    )

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from pdf_extractor import extract_text_from_pdf
from classifier import classify_document, get_document_label
from analyser import analyse_document, get_urgency_color
from audio_generator import generate_audio, build_audio_script, translate_text


URGENCY_EMOJI = {"HIGH": "🔴", "MEDIUM": "🟠", "LOW": "🟢"}
SUPPORTED_LANGUAGES = ["Hindi", "Marathi", "Tamil", "Telugu"]


def format_analysis_html(analysis: dict, doc_label: str) -> str:
    """Convert the analysis dict into a clean HTML display."""
    if "error" in analysis:
        return f"<p style='color:red'>Analysis failed: {safe_text(analysis.get('error', 'Unknown error'))}</p>"

    urgency = analysis.get("urgency_level", "MEDIUM").upper()
    urgency_color = get_urgency_color(urgency)
    urgency_emoji = URGENCY_EMOJI.get(urgency, "🟠")

    html = f"""
        <div style="font-family: sans-serif; max-width: 800px;">

            <div class="doc-type-card">
                <div class="doc-type-title">Document Type</div>
                <div class="doc-type-value">{doc_label}</div>
            </div>

      <div style="background:{urgency_color}18; border:1.5px solid {urgency_color}; border-radius:10px; padding:16px; margin-bottom:16px;">
        <div style="font-size:13px; font-weight:600; color:{urgency_color}; margin-bottom:6px;">
          {urgency_emoji} URGENCY: {urgency}
        </div>
        <div style="font-size:14px; color:#333;">{analysis.get("urgency_reason", "")}</div>
      </div>

      <div style="background:#fff; border:1px solid #e0e0e0; border-radius:10px; padding:16px; margin-bottom:16px;">
        <div style="font-size:13px; color:#666; font-weight:600; margin-bottom:8px;">WHAT THIS DOCUMENT SAYS</div>
        <div style="font-size:15px; color:#222; line-height:1.7;">{analysis.get("simple_summary", "")}</div>
      </div>
    """

    parties = analysis.get("parties_involved", [])
    if parties:
        html += """<div style="background:#fff; border:1px solid #e0e0e0; border-radius:10px; padding:16px; margin-bottom:16px;">
        <div style="font-size:13px; color:#666; font-weight:600; margin-bottom:8px;">PEOPLE INVOLVED</div>"""
        for p in parties:
            html += f"""<div style="display:flex; gap:12px; margin-bottom:6px;">
          <span style="background:#e8f4fd; color:#1565c0; padding:2px 10px; border-radius:20px; font-size:12px; font-weight:500;">{p.get("role","")}</span>
          <span style="font-size:14px; color:#333;">{p.get("name","Not specified")}</span>
        </div>"""
        html += "</div>"

    dates = analysis.get("critical_dates", [])
    if dates:
        html += """<div style="background:#fff3e0; border:1px solid #ffb74d; border-radius:10px; padding:16px; margin-bottom:16px;">
        <div style="font-size:13px; color:#e65100; font-weight:600; margin-bottom:8px;">IMPORTANT DATES</div>"""
        for d in dates:
            html += f"""<div style="margin-bottom:10px; padding:10px; background:#fff8f0; border-radius:8px;">
          <div style="font-size:14px; font-weight:600; color:#bf360c;">{d.get("date","")}</div>
          <div style="font-size:13px; color:#555;">{d.get("label","")}</div>
          <div style="font-size:12px; color:#888; margin-top:2px;">{d.get("importance","")}</div>
        </div>"""
        html += "</div>"

    conditions = analysis.get("key_conditions", [])
    if conditions:
        html += """<div style="background:#fff; border:1px solid #e0e0e0; border-radius:10px; padding:16px; margin-bottom:16px;">
        <div style="font-size:13px; color:#666; font-weight:600; margin-bottom:8px;">KEY CONDITIONS</div>"""
        for c in conditions:
            html += f'<div style="font-size:14px; color:#333; margin-bottom:6px; padding-left:12px; border-left:3px solid #7c4dff;">• {c}</div>'
        html += "</div>"

    ignored = analysis.get("what_happens_if_ignored", "")
    if ignored:
        html += f"""<div style="background:#ffebee; border:1.5px solid #ef5350; border-radius:10px; padding:16px; margin-bottom:16px;">
        <div style="font-size:13px; color:#c62828; font-weight:600; margin-bottom:6px;">⚠️ IF YOU IGNORE THIS DOCUMENT</div>
        <div style="font-size:14px; color:#333;">{ignored}</div>
      </div>"""

    rights = analysis.get("your_rights", [])
    if rights:
        html += """<div style="background:#e8f5e9; border:1px solid #81c784; border-radius:10px; padding:16px; margin-bottom:16px;">
        <div style="font-size:13px; color:#2e7d32; font-weight:600; margin-bottom:8px;">YOUR LEGAL RIGHTS</div>"""
        for r in rights:
            html += f'<div style="font-size:14px; color:#333; margin-bottom:6px;">✓ {r}</div>'
        html += "</div>"

    steps = analysis.get("next_steps", [])
    if steps:
        html += """<div style="background:#e3f2fd; border:1px solid #64b5f6; border-radius:10px; padding:16px; margin-bottom:16px;">
        <div style="font-size:13px; color:#1565c0; font-weight:600; margin-bottom:8px;">WHAT YOU SHOULD DO NOW</div>"""
        for i, s in enumerate(steps, 1):
            html += f"""<div style="display:flex; gap:10px; margin-bottom:8px; align-items:flex-start;">
          <span style="background:#1565c0; color:white; border-radius:50%; width:22px; height:22px; display:flex; align-items:center; justify-content:center; font-size:12px; font-weight:600; flex-shrink:0;">{i}</span>
          <span style="font-size:14px; color:#333;">{s}</span>
        </div>"""
        html += "</div>"

    html += "</div>"
    return html


def translate_analysis(analysis: dict, language: str) -> dict:
    """Translate analysis content to the selected language for display/audio."""
    translated = {}
    for key, value in analysis.items():
        if isinstance(value, str):
            translated[key] = translate_text(value, language)
        elif isinstance(value, list):
            new_list = []
            for item in value:
                if isinstance(item, dict):
                    new_item = {
                        k: translate_text(v, language) if isinstance(v, str) else v
                        for k, v in item.items()
                    }
                    new_list.append(new_item)
                elif isinstance(item, str):
                    new_list.append(translate_text(item, language))
                else:
                    new_list.append(item)
            translated[key] = new_list
        else:
            translated[key] = value

    translated["urgency_level"] = analysis.get("urgency_level", "MEDIUM")
    return translated


def is_text_usable(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) < 200:
        return False
    alpha_count = sum(ch.isalpha() for ch in stripped)
    ratio = alpha_count / max(1, len(stripped))
    return ratio >= 0.2


def process_document(pdf_file, language):
    """Main pipeline: PDF → classify → analyse → display."""
    if pdf_file is None:
        return "<p style='color:red'>Please upload a PDF file.</p>", None, ""

    try:
        text = safe_text(extract_text_from_pdf(pdf_file))
    except ValueError as e:
        return f"<p style='color:red'>{str(e)}</p>", None, ""

    if not is_text_usable(text):
        return (
            "<p style='color:red'>The PDF text looks unreadable (likely a scanned image or corrupted extraction). "
            "Please upload a text-based PDF or run OCR and try again.</p>",
            None,
            "",
        )

    doc_type = classify_document(text)
    doc_label = get_document_label(doc_type)

    try:
        analysis = analyse_document(
            safe_text(text),
            safe_text(doc_type),
            safe_text(doc_label),
        )
    except Exception as e:
        return f"<p style='color:red'>Analysis failed: {str(e)}</p>", None, ""

    translated_analysis = translate_analysis(analysis, language)
    html_output = format_analysis_html(translated_analysis, doc_label)

    audio_script = safe_text(
        build_audio_script(translated_analysis, analysis.get("urgency_level", "MEDIUM"))
    )
    try:
        audio_path = generate_audio(audio_script, language)
    except Exception as e:
        audio_path = None

    status = f"Document analysed: {doc_label} | Language: {language}"
    return html_output, audio_path, status


with gr.Blocks(
    title="Nyaya Mitra — AI Legal Assistant",
    theme=gr.themes.Soft(),
        css="""
        .gradio-container { max-width: 900px !important; margin: auto; }
        #header { text-align: center; padding: 20px 0 10px; }
        #header h1 { font-size: 2rem; color: #1a237e; margin: 0; }
        #header p { color: #555; font-size: 15px; margin: 6px 0 0; }

        .doc-type-card {
            background: linear-gradient(135deg, #111827 0%, #0b1220 100%);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 16px;
            box-shadow: 0 6px 18px rgba(2, 6, 23, 0.25);
        }

        .doc-type-title {
            font-size: 12px;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            color: rgba(255, 255, 255, 0.75);
            margin-bottom: 6px;
        }

        .doc-type-value {
            font-size: 18px;
            font-weight: 600;
            color: #ffffff;
        }
        """
) as app:

    gr.HTML("""
    <div id="header">
      <h1>⚖️ Nyaya Mitra</h1>
      <p>Upload any legal document — get a plain-language explanation and audio in your language</p>
      <p style="font-size:13px; color:#888;">Supports: Court notices, FIRs, Rent agreements, Property documents, Legal notices</p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            pdf_input = gr.File(
                label="Upload Legal Document (PDF)",
                file_types=[".pdf"],
                
            )
            language_input = gr.Dropdown(
                choices=SUPPORTED_LANGUAGES,
                value="Hindi",
                label="Select Audio Language",
            )
            submit_btn = gr.Button(
                "Analyse Document",
                variant="primary",
                size="lg",
            )
            status_text = gr.Textbox(
                label="Status",
                interactive=False,
                visible=True,
            )

        with gr.Column(scale=2):
            analysis_output = gr.HTML(
                label="Document Analysis",
                value="<p style='color:#aaa; text-align:center; margin-top:40px;'>Upload a PDF to see the analysis here</p>"
            )

    gr.HTML("<hr style='margin:20px 0; border-color:#eee;'>")
    gr.HTML("<div style='text-align:center; font-size:14px; color:#555; margin-bottom:8px;'>Audio Summary in your language</div>")

    audio_output = gr.Audio(
        label="Listen to the summary",
        
        interactive=False,
    )

    submit_btn.click(
        fn=process_document,
        inputs=[pdf_input, language_input],
        outputs=[analysis_output, audio_output, status_text],
    )

    gr.HTML("""
    <div style="text-align:center; margin-top:20px; font-size:12px; color:#aaa;">
      Nyaya Mitra — Legal aid for every Indian citizen | Built with Groq + Llama 3 + gTTS
    </div>
    """)


if __name__ == "__main__":
    app.launch(share=True)
