<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8" />
  <title>Chatbot RAG - Consulta à EC 103/2019</title>
  <style>
    body {
      font-family: Arial, sans-serif;
      margin: 40px;
      background-color: #f7f7f7;
      color: #333;
      line-height: 1.6;
    }
    .container {
      background-color: #fff;
      padding: 30px;
      border-radius: 10px;
      box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    h1 {
      color: #005580;
    }
    a.button {
      display: inline-block;
      padding: 10px 20px;
      margin-top: 10px;
      margin-right: 10px;
      color: white;
      background-color: #007ACC;
      text-decoration: none;
      border-radius: 5px;
    }
    a.button:hover {
      background-color: #005f99;
    }
    ol {
      padding-left: 20px;
    }
  </style>
</head>
<body>
  <div class="container">
    <h1>Chatbot com RAG – Consulta à EC 103/2019</h1>

    <p><strong>Descrição:</strong> Este projeto implementa um chatbot baseado em <strong>RAG (Retrieval-Augmented Generation)</strong> para responder perguntas sobre a <strong>Nota Técnica SEI nº 12212/2019</strong>, que analisa os efeitos da <strong>Reforma da Previdência (EC 103/2019)</strong> nos Regimes Próprios de Previdência Social (RPPS) dos entes subnacionais.</p>

    <p>O objetivo é facilitar o acesso às orientações normativas, oferecendo respostas contextualizadas em linguagem natural, com base no conteúdo oficial.</p>

    <h2>Link de Acesso</h2>
    <p>
     📂 <a class="button" href="https://github.com/lureinaldo/chatbot_ec103" target="_blank">Ver o Código no GitHub</a><br><br>
    </p>

    <h2>Como utilizar o projeto</h2>
    <ol>
      <li>Acesse o notebook pelo link do Google Colab.</li>
      <li>Execute as células de código passo a passo.</li>
      <li>Digite sua pergunta em linguagem natural no campo de teste.</li>
      <li>O chatbot retornará uma resposta fundamentada na Nota Técnica.</li>
    </ol>

    <h2>Sobre a Solução</h2>
    <p>O chatbot utiliza técnicas modernas de IA para extrair trechos relevantes do documento oficial e gerar uma resposta coerente e fundamentada. Foi implementado com as bibliotecas LangChain, HuggingFace, FAISS e OpenAI.</p>

    <h2>Autoria</h2>
    <p>Desenvolvido por: <strong>Luciana Reinaldo</strong><br>
    Curso: MBA em Ciência de Dados e Inteligência Artificial Aplicada<br>
    Disciplina: Inteligência Artificial Generativa</p>
  </div>
</body>
</html>
