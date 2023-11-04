#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 11:11:45 2023

@author: sebastianpedraza

Optimizacion en una variable
"""


import matplotlib.pyplot as plt
import numpy as np

class OptimizerManager:
    '''
    Descripcion: Se crea la clase donde vamos a almacenar los
    metodos de optimizacion, ademas de los metodos para agregarlos
    y ejecuarlos sobre la funcion objetivo
    '''
    def __init__(self):
        self.optimizers = {}
        self.algorithm = None

    def register_optimizer(self, name, optimizer_function):
        self.optimizers[name] = optimizer_function

    def optimize(self, algorithm_name, *args, **kwargs):
        if algorithm_name in self.optimizers:
            optimizer_function = self.optimizers[algorithm_name]
            return optimizer_function(*args, **kwargs)
        else:
            raise ValueError(f"El metodo '{algorithm_name}' no esta registrado")



    def GraphObjectiveFunction(self):
        '''
        Descripcion: Funcion para graficar la funcion objetivo.

        Input: None

        Output: Grafica (plot) -> Se muestra la grafica de la funcion obejtivo en el intervalo que la vamos a trabajar, la funcion
        objetivo esta pre-definida en el metodo U(T), del mismo modo los limites con func_limit
        '''

        a, b = self.func_limit()[0], self.func_limit()[1]
        T = np.linspace(a, b, 100)
        U = [self.U(t) for t in T]

        plt.figure(figsize=(6, 3))
        plt.plot(T, U, 'b')
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.grid()
        plt.show()


    def func_limit(self):
        '''
        Descripcion: Funcion definimos los limites [a, b] donde se va a trabajar la funcion objetivo

        Input: None

        Output: limites de integracion (list) -> Se obtiene el primer intervalo sobre el cual vamos a delimitar la funcion obejtivo
        '''

        return [-4, 4]




    def U(self, T: float):

        '''
        Descripcion: Funcion para evaluar la funcion objetivo en un punto x, recuerde que U(T) es la funcion objetivo que usamos para encontrar el minimo

        Input: T(float) ->  T que es el punto donde se va a evaluar la funcion objetivo

        Output: U{T}(float) -> Se devuelve U(T) que es la funcion objetivo evaluada en el punto T
        '''

        return T**4 - T**2 -4*T

    def derU(self, T: float):
        return 4*(T**3) -2*T - 4

    def der2U(self, T):
        return 36*(T**2) + 24*T

    ## Metodo de newton
    def newtonSearch(self):
        epsilon   = 0.0001
        lambda_k  = -2
        k         = 1

        while True:
            if abs(self.derU(lambda_k)) < epsilon:
                print("-------------------------------------------------------")
                print("Punto óptimo {}".format(lambda_k))
                break
            else:
                lambda_k_1 = lambda_k - (self.derU(lambda_k) / self.der2U(lambda_k))
                if abs(lambda_k_1 - lambda_k) < epsilon:
                    print("-------------------------------------------------------")
                    print("Punto óptimo {}".format(lambda_k_1))
                    break
                else:
                    lambda_k = lambda_k_1
            k += 1


    ## Metodo de la seccion dorada
    def GoldenSectionSearch(self):
        '''
        Funcionamiento: Sea  [a,b] el intervalo incial de busqueda, alpha la constante
        sobre la cual calculamos lambda_k y miu_k, h es la constante
        sobre la cual definimos el espacio entre el intervalo encontrado
        '''
        a       =  self.func_limit()[0]
        b       =  self.func_limit()[1]
        alpha   =  (5**(1/2)-1)/2
        h       =  0.001

        '''
        Contador solo lo usamos para mostrar el numero de iteraciones que hemos realizado
        data guarda los valores de el numero de iteracion, lambda y f(lambda)
        '''

        cont = 0

        while(True):

            'calculamos alpha 1 y miu'
            cont    =  cont + 1
            lambda1 =  a + (1-alpha) * (b-a)
            miu1    =  a + alpha*(b-a)

            'calulmamos f(alpha1) y f(alpha2)'
            U_lambda1 = self.U(lambda1)
            U_miu1 = self.U(miu1)

            'como f(lambda) > f(miu), entonces el limite sobre el intervalo [a,b] pasa a ser'
            '[lambda, b]'
            if(U_lambda1 > U_miu1):
                # print("It: {:02d} - a_k: {:.10f} - b_k: {:.10f}  - l_k {:.10f}  - u_k{:.10f}  - f(l_k) {:.10f}  - f(u_k) {:.10f} - b_k-a_k {:.10f}".format(cont, a, b, lambda1, miu1, U(lambda1), U(miu1), b-a))
                a = lambda1

                'de otro modo el limite se reduce el intervalo [a, miu]'
            else:
                # print("It: {:02d} - a_k: {:.10f} - b_k: {:.10f}  - l_k {:.10f}  - u_k{:.10f}  - f(l_k) {:.10f}  - f(u_k) {:.10f} - b_k-a_k {:.10f}".format(cont, a, b, lambda1, miu1, U(lambda1), U(miu1), b-a))

                b = miu1

            'Mostramos numero de iteracion y el intervalo al cual vamos reduciendo'
            print("It: {:02d} - a_k: {:.10f} - b_k: {:.10f}".format(cont, a, b))

            if(np.abs(a - b) < h):

                'Mostramos el intervalo final luego de que este cumpla con el maximo'
                'de distancia entre los puntos a y b sea menos a h'

                print("-------------------------------------------------------")
                print("It: {:02d} - a_k: {:.10f} - b_k: {:.10f}".format(cont, a, b))

                'Como la solucion optima pertenece al intervalo [a,b]'
                'retornamos (a+b) / 2'

                print("(a_k + b_k)/2 == {}".format((a + b)/2))
                break


    ## Metodo de biseccion
    def bisectionSearch(self):
        '''
        Funcionamiento: Sea [a,b] el intervalo incial de busqueda,h es la constante
        sobre la cual definimos el espacio entre el intervalo (intervalo de incertidumbre),
        cont es el contador para mostar el numero de iteracion
        '''
        a    =  self.func_limit()[0]
        b    =  self.func_limit()[1]
        h    =  0.0001
        cont =  0
        while True:
            cont += 1


            if b-a < h:

                'Mostramos el intervalo final luego de que este cumpla con el maximo'
                'de distancia entre los puntos a y b sea menos a h'

                print("-------------------------------------------------------")
                print("It: {:02d} - a_k: {:.10f} - b_k: {:.10f}".format(cont, a, b))

                'Como la solucion optima pertenece al intervalo [a,b]'
                'retornamos (a+b) / 2'
                print("(a_k + b_k)/2 == {}".format((a + b)/2))
                break

            'Calculamos lambda_k'
            lambda_k = (a+b)/2

            'Como el algoritmo se aplica sobre funciones pseudoconvexa'
            'si la derivada en le punto lambda_k es 0, entonces estamos en le minimo'

            if self.derU(lambda_k) == 0:
                print("-------------------------------------------------------")
                print("Solucion optima == {}".format(lambda_k))


                'Si la derivada es >0 evaluada en lambda_k como es una funcion pseudoconvexa'
                'el nuevo intervalo es [a, lambda_k]'
            elif self.derU(lambda_k) > 0:
                # print("It: {:02d} - a_k: {:.10f} - b_k: {:.10f} - l_k: {:.10f} - f(l_k): {:.10f} - a_k-b_k: {:.10f}".format(cont, a, b, lambda_k, derU(lambda_k), b-a))
                b = lambda_k


                'Si la derivada es <0 evaluada en lambda_k como es una funcion pseudoconvexa'
                'el nuevo intervalo es [lambda_k, b]'
            else:
                # print("It: {:02d} - a_k: {:.10f} - b_k: {:.10f} - l_k: {:.10f} - f(l_k): {:.10f} - a_k-b_k: {:.10f}".format(cont, a, b, lambda_k, derU(lambda_k), b-a))
                a = lambda_k
            print("It: {:02d} - a_k: {:.10f} - b_k: {:.10f}".format(cont, a, b))


    # Metodo de Fibonacci

    def fibonacci(self,n):
        '''
        Funcionamiento: Fibonnaci retorna el resultado de la sucesion de fibonacci usado en el metodo de busqueda
        para encontrar  el numero de iteraciones que vamos a hacer para encontrar el
        intervalo donde esta la  solucion optima
        '''

        if n<= 1: return 1
        else: return self.fibonacci(n-1) + self.fibonacci(n-2)

    def iterationFinder(self,a,b,h):
        cv = (b-a)/h
        i = 0
        while True:
            if self.fibonacci(i) > cv: return i
            i+=1


    def fibonacciSearch(self):
        '''
        Funcionamiento: Sea[a,b] el intervalo incial de busqueda,h la longitud final
        de incertidumbre y cont para llevar contador para mostar el numero de iteracion,
        e que es un epsilon > 0 que es un constante de distinguibilidad que cumple con
        e < h y n la cantidad de iteraciones
        '''

        a    =  self.func_limit()[0]
        b    =  self.func_limit()[1]
        e    =  0.01
        h    =  0.1
        cont =  1
        n    =  self.iterationFinder(a, b, h)

        while True:

            'Calculamos lambda_k y miu_k, y llevamos el contador de iteraciones'
            lambda_k =  a + (1-(self.fibonacci(n-1)/self.fibonacci(n)))*(b-a)
            miu_k    =  a + (self.fibonacci(n-1)/self.fibonacci(n))*(b-a)
            cont    +=  1

            'si f(lambda_k) > f(miu_k) ajustamos el valor del intervalo a [lambda_k, b_k]'
            if self.U(lambda_k) > self.U(miu_k):
                # print("It: {:02d} - a_k: {:.10f} - b_k: {:.10f}  - l_k {:.10f}  - u_k{:.10f}  - f(l_k) {:.10f}  - f(u_k) {:.10f} - b_k-a_k {:.10f}".format(cont, a, b, lambda_k, miu_k, U(lambda_k), U(miu_k), b-a))
                a = lambda_k

                'si f(lambda_k) <= f(miu_k) ajustamos el valor del intervalo a [a_k, miu_k]'
            else:
                # print("It: {:02d} - a_k: {:.10f} - b_k: {:.10f}  - l_k {:.10f}  - u_k{:.10f}  - f(l_k) {:.10f}  - f(u_k) {:.10f} - b_k-a_k {:.10f}".format(cont, a, b, lambda_k, miu_k, U(lambda_k), U(miu_k), b-a))
                b = miu_k

            print("It: {:02d} - a_k: {:.10f} - b_k: {:.10f}".format(cont, a, b))


            ## condicional para cuando llegamos a la iteracion n-2
            if cont == n-2: break


        miu_k = lambda_k + e

        '''
        Mostramos el intervalo final luego de que este cumpla con el maximo
        de distancia entre los puntos a y b sea menos a h
        '''

        if self.U(lambda_k) > self.U(miu_k):
            # print("It: {:02d} - a_k: {:.10f} - b_k: {:.10f}  - l_k {:.10f}  - u_k{:.10f}  - f(l_k) {:.10f}  - f(u_k) {:.10f} - b_k-a_k {:.10f}".format(cont, a, b, lambda_k, miu_k, U(lambda_k), U(miu_k), b-a))
            a = lambda_k

            "si f'(lambda_k) <= f'(miu_k) ajustamos el valor del intervalo a [a_k, miu_k]"
        else:
            # print("It: {:02d} - a_k: {:.10f} - b_k: {:.10f}  - l_k {:.10f}  - u_k{:.10f}  - f(l_k) {:.10f}  - f(u_k) {:.10f} - b_k-a_k {:.10f}".format(cont, a, b, lambda_k, miu_k, U(lambda_k), U(miu_k), b-a))
            b = miu_k


        print("-------------------------------------------------------")
        '''
        Como la solucion optima pertenece al intervalo [a,b]
        retornamos (a+b) / 25
        '''
        print("(a_k + b_k)/2 == {}".format((a + b)/2))




'''
Notas:

Algunos ejemplos con los que se pueden probar para seccion dorada y fibonacci use
Funcion: T**4 + 3*(T**3) + T**2 + 2*T + 4
limites: -4, 2

Nota: Para estos dos metodos no necesitamos definir correctamente derU ni der2U porque
estos metodos no las usan


Para biseccion use
Funcion: T**4 - T**2 -4*T
Primera derivada: 4*(T**3) -2*T - 4
limites: -4, 4

Nota: Para este metodo solo es necesario la funcion y la primera derivada


Para newton considere usar:
Funcion: 3*(T**4) + 4*(T**3)
Primera derivada: 12*(T**3) + 12*(T**2)
Segunda derivada: 36*(T**2) + 24*T
con lambda = -2 o lambda = 2
'''


# Crear una instancia de la clase OptimizerManager y registra los metodos de optimizacion
optimizer_manager = OptimizerManager()
optimizer_manager.register_optimizer("GoldenSection", optimizer_manager.GoldenSectionSearch)
optimizer_manager.register_optimizer("Newton", optimizer_manager.newtonSearch)
optimizer_manager.register_optimizer("Bisection", optimizer_manager.bisectionSearch)
optimizer_manager.register_optimizer("Fibonacci", optimizer_manager.fibonacciSearch)




## Definimos el metodo que vamos a usar
optimizer_manager.algorithm = "GoldenSection"

# Utilizamos el metodo de optimización y el gráfico
optimizer_manager.optimize(optimizer_manager.algorithm)
optimizer_manager.GraphObjectiveFunction()
