import os
import gradio as gr
from llama_cpp import Llama

# Configurações do Modelo
REPO_ID = "HauhauCS/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive"
FILENAME = "Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-IQ3_M.gguf"

print("🤖 Inicializando e baixando o modelo Gemma-4-E4B...")
# Inicializa o modelo com suporte a GPU (-1 descarrega tudo na GPU)
# Usando n_ctx=2048 para um bom contexto e n_gpu_layers=-1 se GPU disponível
llm = Llama.from_pretrained(
    repo_id=REPO_ID,
    filename=FILENAME,
    n_ctx=2048,
    n_gpu_layers=-1  # Coloca todas as camadas na GPU
)
print("✅ Modelo carregado com sucesso!")

def chat_respond(message, history, system_prompt, temperature, max_tokens, top_p, repeat_penalty):
    # Organiza a lista de mensagens para o modelo
    messages = []
    
    # Adiciona o System Prompt se o usuário definiu algum
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt})
    
    # Histórico de conversas formatado
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})
    
    # Nova mensagem do usuário
    messages.append({"role": "user", "content": message})
    
    print(f"💬 Usuário: {message}")
    print(f"⚙️ Configs: Temp={temperature}, MaxTokens={max_tokens}, TopP={top_p}")
    
    try:
        # Gera a resposta via streaming para dar um efeito premium em tempo real
        response_stream = llm.create_chat_completion(
            messages=messages,
            temperature=float(temperature),
            max_tokens=int(max_tokens),
            top_p=float(top_p),
            repeat_penalty=float(repeat_penalty),
            stream=True
        )
        
        partial_text = ""
        for chunk in response_stream:
            delta = chunk['choices'][0]['delta']
            if 'content' in delta:
                partial_text += delta['content']
                yield partial_text
                
    except Exception as e:
        yield f"⚠️ Ocorreu um erro na geração: {str(e)}"

# ==========================================================================
# INTERFACE GRÁFICA GRADIO (BLOCS)
# ==========================================================================
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="indigo")) as demo:
    
    # Cabeçalho
    with gr.Row():
        with gr.Column(scale=10):
            gr.Markdown(
                """
                # 🚀 LOGI-LLM: Gemma-4-E4B Uncensored
                ### Interface de chat interativa de alta performance rodando na GPU do Colab!
                """
            )
            
    # Layout de duas colunas (Painel de controle à esquerda, Chat à direita)
    with gr.Row():
        
        # Painel de Parâmetros (Sidebar)
        with gr.Column(scale=3, variant="panel"):
            gr.Markdown("### ⚙️ Parâmetros do Modelo")
            
            system_prompt = gr.Textbox(
                value="Você é um assistente de IA prestativo, inteligente e responde de forma clara em Português.",
                label="System Prompt (Personalidade)",
                placeholder="Ex: Você é um poeta focado em logística...",
                lines=4
            )
            
            with gr.Accordion("⚙️ Ajustes Avançados", open=True):
                temperature = gr.Slider(
                    minimum=0.1, maximum=1.5, value=0.7, step=0.05,
                    label="Temperatura (Criatividade)",
                    info="Valores altos deixam a IA mais criativa, baixos mais precisa."
                )
                max_tokens = gr.Slider(
                    minimum=64, maximum=2048, value=512, step=64,
                    label="Max Tokens (Tamanho da resposta)",
                    info="Tamanho limite de palavras/tokens gerados por resposta."
                )
                top_p = gr.Slider(
                    minimum=0.1, maximum=1.0, value=0.9, step=0.05,
                    label="Top P (Nucleus Sampling)",
                    info="Controla a diversidade de palavras consideradas."
                )
                repeat_penalty = gr.Slider(
                    minimum=1.0, maximum=1.5, value=1.1, step=0.05,
                    label="Penalidade de Repetição",
                    info="Impede a IA de repetir frases ou ideias semelhantes."
                )
                
            gr.HTML("<hr style='border: 0; border-top: 1px solid rgba(255,255,255,0.1); margin: 15px 0;'>")
            gr.Markdown(
                """
                🤖 **Modelo**: `Gemma-4-E4B`
                💾 **Formato**: GGUF (IQ3_M)
                ⚡ **Aceleração**: CUDA (GPU)
                """
            )

        # Painel do Chat
        with gr.Column(scale=7):
            chatbot = gr.ChatInterface(
                fn=chat_respond,
                additional_inputs=[system_prompt, temperature, max_tokens, top_p, repeat_penalty],
                chatbot=gr.Chatbot(height=550, placeholder="🤖 **Como posso te ajudar hoje? Faça qualquer pergunta!**"),
                textbox=gr.Textbox(placeholder="Digite sua mensagem aqui...", container=False, scale=7),
                submit_btn="Enviar 🚀",
                retry_btn="Tentar Novamente 🔄",
                undo_btn="Desfazer ↩️",
                clear_btn="Limpar Chat 🗑️",
            )

# Lança a interface
if __name__ == "__main__":
    # share=True é fundamental para gerar o link público acessível fora do Colab!
    demo.queue().launch(share=True, debug=True)
