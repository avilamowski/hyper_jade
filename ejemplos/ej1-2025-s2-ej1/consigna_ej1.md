[1 tool called]

## Consigna

Se desea escribir un programa que, en base a datos de alumnos y notas, visualice información valiosa de la cursada.

### Firma requerida
`mostrar_cursada(nombres, apellidos, notas1p, notas2p)`

- `nombres`: lista de strings con los nombres de los alumnos.
- `apellidos`: lista de strings con los apellidos de los alumnos.
- `notas1p`: lista de floats (0-10) para el primer parcial.
- `notas2p`: lista de floats (0-10) para el segundo parcial.

### Requisitos

1. **Alumnos que recursan (1.5 ptos)**
   - Promedio de ambos parciales ≤ 3.
   - Mostrar `Nombre Apellido - Promedio: X.XX` redondeado a dos decimales.
   - Cada tres alumnos imprimir una barra de separación (`----------------------------`).

2. **Apellidos más largos (1 pto)**
   - Mostrar las notas del segundo parcial de los alumnos con el apellido más largo.
   - Si hay empates, mostrar todos.

3. **Posible promoción (1.5 ptos)**
   - Promedio > 7 pero uno de los parciales está entre 6 (inclusive) y 7 (exclusivo).
   - Mostrar `Nombre Apellido` ordenados alfabéticamente por nombre en orden descendente.

### Consideraciones

- Seguir estrictamente los formatos de los ejemplos.
- Las listas tienen igual longitud (pueden ser vacías).  
  Si las cuatro listas están vacías, imprimir solo una vez `No hay notas cargadas`.
- Mantener modularidad: una función puede invocar otras funciones.

### Ejemplo 1

```python
nombres = ["Ana", "Juan", "Luis", "María", "Lucía", "Ruben", "Adrian", "Jorge"]
apellidos = ["Li", "Gómez", "Paz", "Sosa", "Ro", "Paz", "Martinez", "Carranza"]
notas1p = [1, 2, 0, 8, 6.50, 9, 9, 1]
notas2p = [1, 1.50, 3, 7, 9, 7, 10, 5]

mostrar_cursada(nombres, apellidos, notas1p, notas2p)
```

Salida esperada:

```
Alumnos que recursan la materia:
Ana Li - Promedio: 1.0
Juan Gómez - Promedio: 1.75
Luis Paz - Promedio: 1.5
----------------------------
Jorge Carranza - Promedio: 3.0

Notas del 2do parcial de apellidos más largos
Martinez - Nota 2P: 10
Carranza - Nota 2P: 5

Alumnos que se debe revisar posible promoción:
Lucía Ro
```

### Ejemplo 2

```python
mostrar_cursada([], [], [], [])
```

Salida esperada:

```
No hay notas cargadas
```

## Criterios de corrección

1. Recursantes: cálculo correcto del promedio.
2. Recursantes: detección correcta de recursantes.
3. Recursantes: formato esperado de la salida.
4. Recursantes: impresión de barra cada tres alumnos.
5. Segundo parcial: identificación de apellidos más largos.
6. Segundo parcial: correspondencia correcta notas-apellidos.
7. Segundo parcial: formato esperado de la salida.
8. Posible promoción: identificación correcta del caso.
9. Posible promoción: agregado correcto a la lista de salida.
10. Posible promoción: orden y formato al imprimir.
11. No resuelve la consigna.
12. Otros errores (ejecución, ciclos infinitos, indentación, uso de estructuras no vistas).