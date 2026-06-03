# Configuração de Ambiente Python no VSCode
 
## 1. Criar um ambiente virtual (venv)
 
Na pasta do projeto, execute:
 
```bash
python -m venv .venv
```
 
Caso o comando `python` não funcione:
 
```bash
py -m venv .venv
```
 
---
 
## 2. Ativar o ambiente virtual
 
**Windows (CMD / PowerShell)**
 
```bash
.venv\Scripts\activate
```
 
Se ativou corretamente, o terminal exibirá algo como:
 
```
(.venv)
```
 
---
 
## 3. Instalar dependências
 
```bash
pip install pandas requests
```
 
---
 
## 4. Selecionar o interpretador Python no VSCode
 
Abra a paleta de comandos:
 
```
Ctrl + Shift + P
```
 
Digite:
 
```
Python: Select Interpreter
```
 
Selecione o interpretador referente ao `.venv`.
 
---
 
## 5. Verificar se tudo está funcionando
 
**Conferir bibliotecas instaladas**
 
```bash
pip list
```