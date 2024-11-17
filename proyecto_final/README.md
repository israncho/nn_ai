# Proyecto Final de Redes Neuronales

## Configuracion
### Instalacion de requerimientos

```bash
pip install --upgrade -r requirements.txt
```

Registro del entorno virtual

```bash
python -m ipykernel install --user --name=nn_ai_venv --display-name "Python (nn_ai_venv)"
```

### Configuracion del API key de Kaggle

Una vez que ya esta en tu sistema el archivo

```bash
kaggle.json
```

Se deben ejecutar los siguientes comandos

```bash
mkdir -p ~/.kaggle
mv /ruta/donde/descargaste/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

Y para probar que esto funciono ejecuta:

```bash
kaggle datasets list
```
