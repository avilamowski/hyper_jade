def es_letra(c):
  es=True
  letras="ABCDEFGHIJKLMNÑOPQRSTUVXYZabcdefghijklmnopqrstuvxyz"
  if c not in letras:
    es=False
  return es
 
 
def es_numero(c):
  es=True
  numeros="0123456789"
  if c not in numeros:
    es=False
  return es
    
def es_espacio(c):
  es=True
  espacio=" "
  if not(c==espacio):
    es=False
  return es
    
def es_salto_de_linea(c):
  es=True
  salto="\n"
  if not(c=="\n"):
    es=False
  return es
    
def hay_texto(texto):
  es=True
  vacio=""
  if texto==vacio:
    es=False
  return es
  
def validar(texto):
  valido=True
  if not(hay_texto(texto)):
      valido=False
  else:
    for caracter in texto:
      if not(es_letra(caracter) or es_numero(caracter) or es_espacio(caracter) or es_salto_de_linea(caracter)):
        valido=False
        print("Texto invalido")
  return valido
 
 
 
 
def split_custom(texto, separador="\n"):
    resultado = []
    palabra = ""
    for char in texto:
      if char == separador:
        if palabra:
          resultado.append(palabra)
          palabra = ""
      else:
            palabra += char
    if palabra:
      resultado.append(palabra)
  return resultado
    
def strip_custom(texto):
  inicio, fin = 0, len(texto) -1
  while inicio <= fin and texto[inicio] == " ":
      inicio += 1
  while fin >= inicio and texto[fin] == " ":
      fin -= 1
  return texto[inicio:fin+1]
 
 
 
 
def borrar_vacias(lista):
  lista_sin_vacios=[]
  for i in range(len(lista)):
    elemento=lista[i]
    valido=False
    if not(es_espacio(elemento)):
      lista_sin_vacios.append(i, elemento)
    
  return lista_sin_vacios
  
 
 
def lista_lineas(lista):
  for i in range(len(lista)):
    valido=False
    for j in range (len(lista[i])):
      if not(lista[i][j]=="\n"):
        valido=True
    if valido:    
      print(f"{i+1}:{lista[i]}")
  return lista
        
def sacar_espacios(lista):
  for elemento in range(len(lista)):
    strip_custom(lista)
  return lista
 
 
 
 
 
 
def mostrar_lineas(texto):
  try:
    validar(texto):
      if validar:
        lista_lineas(borrar_vacias(sacar_espacios(strip_custom(split_custom(sigo)))))
        return
      else:
        return 
  
def promedio_numeros(texto):
  if validar(texto):
    suma=0
    total=0
    for caracter in texto:    #aca quiero que recorra todo el texto, y si encuentra un numero que se fije el caracter siguiente hasta que no sea un numero
      numero_del_momento=0
        while es_numero(texto[caracter+1])
          for j in range(caracter+2)
        numero_del_momento=texto[caracter,j]
        suma+=numero_del_momento
        total+=1
    try:
      promedio=suma/totla
      return f"El promedio de los numeros en el texto es de {promedio}"
    except ZeroDivisionError:
      return "No se ingresaron números en el texto"
#se asume una lista donde cada elemento es una palabra
def cantidad_palabras(split_custom(lista)," "):
  lista_de_unicas=[]
  for palabra in lista:
    if not(es_numero(palabra)):
      if palabra not in lista_de_unicas:
        lista_de_unicas.append(palabra)
  return lista_de_unicas
 
 
def mayor_a_menor(lista):
    ln = len(lista)
    for i in range(ln):   
      for j in range(ln-i-1):
        ln1=len(i)
        ln2=len(j)
        if( ln1<ln2 ):
          ln1, ln2 = ln1, ln2
  return lista
 
 
def sin_repetir(texto):
  if validar(texto):
    try: 
      a_escribir=cantidad_palabras(mayor_a_menor(sin_repetir(texto)))
      with open("sin_repeticiones.txt","w")encoding="utf-8" as archivo
      for caracter in a_escribir:
        archivo.write(caracter)
      archivo.close()
      return ""
    except FileNotFoundError:
      return ""
    