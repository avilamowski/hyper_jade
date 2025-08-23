
 
texto='Hola  gente del universo\n \n123 \n Chau todos\n 90 adios'
def ordenar_custom(lista):
    for i in range(len(lista)):
        for j in range(i + 1, len(lista)):
            if lista[i] > lista[j]:
                lista[i], lista[j] = lista[j], lista[i]
    return lista
 
def split_custom(texto, separador=" "):
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
    
def es_letra(caracter):
  if not(('a'<=caracter<='z') or ('A'<=caracter<='Z')):
    return False
  else:
    return True 
    
def es numero(elem):
  try:
    int(elem)
    return True
  except ValueError:
    return False
  
def es_numero(caracter):
  if ("0"<=caracter<="9"):
    return True
  else:
    return False
    
def validar(texto):
  if texto:
    valido=True
    for linea in texto:
      for i in range(len(linea)):
        if not ((es_letra(linea[i])) or (es_numero(linea[i])) or (linea[i]==' ') or (linea[i]=='\n')):
          valido=False
    if valido:
      return True
    else:
      return False
        
print(validar(texto))#funciona!!
 
 
def lista_texto(texto): #separo por \n
  lista_texto=split_custom(texto,'\n')
  return lista_texto 
  
def mostrar_lineas(lista_texto):
  pos=1
  for i in range(len(lista_texto)):
    if lista_texto[i]!='\n':
      print(f'{pos}: {lista_texto[i]}')
      pos+=1
      
def promedio(lista_texto, texto):
  if validar_texto(texto):
    suma=0
    cant=0
    for i in range(len(lista_textos)):
      if es_numero:
        suma+=int(lista[i])
        cant+=1
    promedio=(suma/cant)
    print(f'El promedio de los números en el texto es {promdio}')
 
def sin_repetir(texto):
  texto_split= split_custom(texto, ' ')
  sin_repetir=[]
  for i in range(texto_split):
    repetido=False
    for j in range(i+1, len(texto_split)):
      if texto_split[j]==texto_split[i]:
        repetido=True
    if repetido==False:
      sin_repetir.append(texto_split[i])
  retun sin_repetir
 
 
def escrbir_archivo(sin_repetir):
  with open('sin_repeticiones.txt', 'r', encoding="utf8")as file:
    file.write(ordenar_custom(sin_repetir))
 
def main(): #esto no era necesario pero lo hago para guiarme 
  validar(texto)
  lista=lista_texto(texto)
  mostrar_lineas(lista)
  sin_repetir(texto)
  escribir_archivo(sin_repetir)
main()
 