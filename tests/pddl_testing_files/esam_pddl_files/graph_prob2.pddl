(define (problem graph-ver1)
   (:domain graph)
   (:objects u1 u2 u3 u4 u5)
   (:init (vertex u1)
          (vertex u2)
          (vertex u3)
          (vertex u4)
          (vertex u5))

   (:goal (and (has-edge u1 u1) (has-edge u2 u2))))