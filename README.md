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

Caso dê algum erro digite:
'pip install -r requirements.txt'
```

## Separação dos Arquivos

```markdown
📁 Pipeline-ML
│
├── 📁 configs                  
│   └── 📄 test_dataset.jason   # Arquivo do teste do Dataset
├── 📁 data                     
│   └── 📄 test_dataset.csv     # Arquivo do teste do dataset
├── 📁 models                   
│   └── 📄 base_model.py        # Classe abstrata comum a todos os modelos
├── 📁 pipeline                 
│   └── 📄 runner.py            # Orquestra todas as etapas do sistema
├── 📁 src                      
│   ├── 📄 loader.py            # Principal função: Carregar os dados     
│   ├── 📄 prepocessor.py       # Principal função: Aplicar as regras definidas no config.json
│   └── 📄 validadot.py         # Principal função: Validar esquema, tipos e ranges do DataFrame
├── 📁 test
│   ├── 📄 agentTest.py         # Teste do Agente
│   ├── 📄 basemodelTest.py     # Teste do BaseModel
│   ├── 📄 dataloaderTest.py    # Teste do Loader
│   ├── 📄 pipelineTest.py      # Teste do Pipeline
│   ├── 📄 preprocessor.py      # Teste do Preprocessor
│   └── 📄 validatorTest.py     # Teste do Validator
├── 📄 .gitignore               # Arquivo para ignorar outros arquivos
├── 📄 Readme.md                # Documento para explicar o código
├── 📄 reporter.py              # Arquivo para repostar erros e gerar relatorios
├── 📄 requirements.txt         # Requirimentos do pip
└── 📄 main.py                  # Exibe o resultado final 
