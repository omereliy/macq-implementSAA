
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Domain file automatically generated by the Tarski FSTRIPS writer
;;; 
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define (domain logistics)
    (:requirements :typing)
    (:types
        obj - object
        object
    )

    (:constants
        
    )

    (:predicates
        (vertex ?x1 - obj)
        (has-edge ?x1 - obj ?x2 - obj)
    )

    (:functions
        
    )

    

    
    (:action add-edge_1_1
     :parameters (?x0 - obj ?x1 - obj)
     :precondition (and (vertex ?x1) (vertex ?x0))
     :effect (and
        (has-edge ?x1 ?x0)
        (has-edge ?x0 ?x1))
    )


    (:action add-edge_1_2
     :parameters (?x0 - obj ?x1 - obj)
     :precondition (and (vertex ?x1) (vertex ?x0) (has-edge ?x1 ?x0))
     :effect (and
        (has-edge ?x0 ?x1))
    )

)