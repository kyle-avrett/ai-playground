import gradio as gr
from answer import answer_question


def format_context(context):
    sections = ["<h2 style='color: #ff7800;'>Relevant Context</h2>"]
    for doc in context:
        sections.append(
            f"<span style='color: #ff7800;'>Source: {doc.metadata['source']}</span>\n\n"
            f"{doc.page_content}"
        )
    return "\n\n".join(sections)


def text_content(message):
    content = message["content"]
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        )
    return str(content)


def text_history(history):
    return [
        {"role": message["role"], "content": text_content(message)}
        for message in history
    ]


def chat(history):
    last_message = text_content(history[-1])
    answer, context = answer_question(last_message, text_history(history[:-1]))
    history.append({"role": "assistant", "content": answer})
    return history, format_context(context)


def main():
    def put_message_in_chatbot(message, history):
        return "", history + [{"role": "user", "content": message}]

    theme = gr.themes.Soft(font=["Inter", "system-ui", "sans-serif"])

    with gr.Blocks(title="Insurellm Expert Assistant") as ui:
        gr.Markdown("# 🏢 Insurellm Expert Assistant\nAsk me anything about Insurellm!")

        with gr.Row():
            with gr.Column(scale=1):
                chatbot = gr.Chatbot(
                    label="💬 Conversation", height=600, buttons=["copy"]
                )
                message = gr.Textbox(
                    label="Your Question",
                    placeholder="Ask anything about Insurellm...",
                    show_label=False,
                )

            with gr.Column(scale=1):
                context_markdown = gr.Markdown(
                    label="📚 Retrieved Context",
                    value="*Retrieved context will appear here*",
                    container=True,
                    height=600,
                )

        message.submit(
            put_message_in_chatbot,
            inputs=[message, chatbot],
            outputs=[message, chatbot],
        ).then(chat, inputs=chatbot, outputs=[chatbot, context_markdown])

    ui.launch(inbrowser=True, theme=theme)


if __name__ == "__main__":
    main()
