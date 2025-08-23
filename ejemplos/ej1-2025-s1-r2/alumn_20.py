#Ejercicio 1
def validar(texto):
  c=0
  while c<len(texto):
   if (ord(texto[c])<=90 and ord(texto[c])>65) or ord(texto[c])==32 or (ord(texto[c]) >= 97 and ord(texto[c])<= 122) or (ord(texto[c])>=48 and ord(texto[c])<=57) or texto[c]=='\n':
     c+=1
   else:
     return False
  if c==len(texto) and texto!='':
    return True
    
texto = 'Holaaa1'
print(validar(texto))
 
#Ejercicio 2
﻿def validar(texto):
  
  c=0
  while c<len(texto):
   if (ord(texto[c])<=90 and ord(texto[c])>65) or ord(texto[c])==32 or (ord(texto[c]) >= 97 and ord(texto[c])<= 122) or (ord(texto[c])>=48 and ord(texto[c])<=57) or texto[c]=='\n':
     c+=1
   else:
     return False
  if c==len(texto) and texto!='':
    return True
def split_custom2(texto, separador="\n"):
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
def mostrar_lineas(texto):
  if validar(texto)==True:
    linea = split_custom2(texto, separador="\n")
    for i in range(len(linea)):
      if linea[i]!=' ':
        print('1:',linea[i])
texto='Hola  gente del universo\n \n123 \n Chau todos\n 90 adios'
mostrar_lineas(texto)  
#Ejercicio 3
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
 
def promedio_numeros(texto):
  resultado = split_custom(texto, separador=" ")
  if validar(texto)==True:
    numero = []
    suma_nums = 0
    cant_numeros = 0
    for t in range(len(resultado)):
      numero.append(0)
    for i in range(len(resultado)):
      for c in range(len(resultado[i])):
        if (ord(resultado[i][c])>=48 and ord(resultado[i][c])<=57):
          numero[i]+=int(resultado[i][c])
          suma_nums+=int(resultado[i][c])
          cant_numeros+=1
    promedio = suma_nums / cant_numeros
    print(promedio)
texto = 'Hola1  gente del universo\n \n123 \n Chau todos\n 90 adios'
promedio_numeros(texto)
 
#Ejercicio 4
 
########