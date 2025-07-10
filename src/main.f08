PROGRAM main
    USE omp_lib
    IMPLICIT NONE
    !> The number of vertices in the graph.
    INTEGER, PARAMETER :: GRAPH_ORDER = 1000
    !> Parameters used in pagerank convergence, do not change.
    REAL(KIND=8), PARAMETER :: DAMPING_FACTOR = 0.85_8
    !> The number of seconds to not exceed forthe calculation loop.
    INTEGER, PARAMETER :: MAX_TIME = 10
    REAL(KIND=8) :: start
    REAL(KIND=8) :: end
    !> The array in which each vertex pagerank is stored.
    REAL(KIND=8), DIMENSION(0:GRAPH_ORDER-1) :: pagerank
    ! Calculates the sum of all pageranks. It should be 1.0, so it can be used as a quick verification.
    REAL(KIND=8) :: sum_ranks = 0.0
    REAL(KIND=8) :: max_diff = 0.0
    REAL(KIND=8) :: min_diff = 1.0
    REAL(KIND=8) :: total_diff = 0.0
    INTEGER :: i
    integer :: io

    !> @brief Indicates which vertices are connected.
    !> @details If an edge links vertex A to vertex B, then adjacency_matrix[A][B]
    !> will be 1.0. The absence of edge is represented with value 0.0.
    !> Redundant edges are still represented with value 1.0.
    REAL(KIND=8), DIMENSION(0:GRAPH_ORDER-1,0:GRAPH_ORDER-1) :: adjacency_matrix

    WRITE(*, '(A,A)') 'This program has two graph generators: generate_nice_graph and generate_sneaky_graph. ', &
        'If you intend to submit, your code will be timed on the sneaky graph, remember to try both.'

    ! Get the time at the very start.
    start = omp_get_wtime()

    !CALL generate_nice_graph()
    CALL generate_sneaky_graph()

    CALL calculate_pagerank(pagerank)

    DO i = 0, GRAPH_ORDER - 1
        IF (MODULO(i, 100) .EQ. 0) THEN
            WRITE(*, '(A,I0,A,F0.6)') 'PageRank of vertex ', i, ': ', pagerank(i)
        END IF
        sum_ranks = sum_ranks + pagerank(i)
    END DO
    WRITE(*, '(A,F0.12,A,F0.12,A,F0.12)') 'Sum of all pageranks = ', sum_ranks, &
        ', total diff = ' , total_diff, ' max diff = ', max_diff, &
        ' and min diff = ', min_diff
    end = omp_get_wtime()

    WRITE(*, '(A,F0.2,A)') 'Total time taken: ', end - start, ' seconds.'

    open(newunit=io, file="pageranks.txt", status="replace", action="write")

    do i = 0, GRAPH_ORDER - 1
        write(io, "(F17.15)") pagerank(i)
    end do

    close(io)

CONTAINS

    SUBROUTINE initialize_graph()
        INTEGER :: i
        INTEGER :: j

        DO i = 0, GRAPH_ORDER - 1
            DO j = 0, GRAPH_ORDER - 1
                adjacency_matrix(j,i) = 0.0
            END DO
        END DO
        RETURN
    END SUBROUTINE

    !> @brief Calculates the pagerank of all vertices in the graph.
    !> @param pagerank The array in which store the final pageranks.
    SUBROUTINE calculate_pagerank(pagerank)
        IMPLICIT NONE

        REAL(KIND=8), DIMENSION(0:GRAPH_ORDER-1) :: pagerank
        REAL(KIND=8), DIMENSION(0:GRAPH_ORDER-1) :: new_pagerank
        REAL(KIND=8) :: pagerank_total
        REAL(KIND=8) :: initial_rank = 1.0_8 / REAL(GRAPH_ORDER, KIND=8);
        REAL(KIND=8) :: damping_value = (1.0_8 - DAMPING_FACTOR) / REAL(GRAPH_ORDER, KIND=8)
        REAL(KIND=8) :: diff = 0.0
        INTEGER(KIND=8) :: iteration = 0
        REAL(KIND=8) :: start
        REAL(KIND=8) :: elapsed
        REAL(KIND=8) :: time_per_iteration = 0.0
        REAL(KIND=8) :: iteration_start
        REAL(KIND=8) :: iteration_end
        INTEGER :: outdegree
        INTEGER :: i
        INTEGER :: j
        INTEGER :: k
        INTEGER :: source_i
        INTEGER :: destination_i
        real(kind=8) :: outdeg_div_arr(0:GRAPH_ORDER-1)
        integer :: tmp_outdeg
        real(kind=8) :: tmp_sum
        integer :: nnonzero
        integer, allocatable :: dest_offset(:)
        integer, allocatable :: dest_map(:)
        real(kind=8) :: new_pr

        ! Initialise all vertices to 1/n.
        DO i = 0, GRAPH_ORDER - 1
            pagerank(i) = initial_rank
        END DO

        start = omp_get_wtime()
        elapsed = omp_get_wtime() - start
        DO i = 0, GRAPH_ORDER - 1
            new_pagerank(i) = 0.0
        END DO

        ! If we exceeded the MAX_TIME seconds, we stop. If we typically spend X
        ! seconds on an iteration, and we are less than X seconds away from
        ! MAX_TIME, we stop.
        !$omp parallel &
        !$omp default(shared) &
        !$omp private(destination_i,k,tmp_outdeg,source_i,i,new_pr)

        ! determine the outdegree of each vertex and store the inverse in outdeg_div_arr
        !$omp do schedule(dynamic)
        do destination_i=0,graph_order-1
            tmp_outdeg = 0
            do k=0,graph_order-1
                if(adjacency_matrix(k,destination_i) == 1) tmp_outdeg=tmp_outdeg+1
            enddo
            if (tmp_outdeg .ne. 0) then
                outdeg_div_arr(destination_i) = 1.0d0/dble(tmp_outdeg)
            else
                outdeg_div_arr(destination_i) = 0.0d0
            end if
        enddo

        ! determine the number of non-zero entries in the adjacency matrix for each source vertex
        !$omp single
        do source_i = 0, GRAPH_ORDER - 1
            do destination_i = 0, GRAPH_ORDER - 1
                if (adjacency_matrix(destination_i,source_i) == 1) nnonzero = nnonzero + 1
            end do
        end do

        allocate(dest_offset(0:GRAPH_ORDER))
        allocate(dest_map(0:nnonzero-1))

        ! populate the destination offset and mapping arrays
        nnonzero = 0
        dest_offset(0) = 0 ! initialise first offset of column 0
        do source_i = 0, GRAPH_ORDER-1
            do destination_i = 0, GRAPH_ORDER-1
                if (adjacency_matrix(destination_i, source_i) == 1) then
                    dest_map(nnonzero) = destination_i
                    nnonzero = nnonzero + 1
                end if
            end do
            dest_offset(source_i+1) = nnonzero
        end do
        !$omp end single


        DO WHILE (elapsed .LT. MAX_TIME .AND. (elapsed + time_per_iteration) .LT. MAX_TIME)

            !$omp single
            iteration_start = omp_get_wtime();
            diff = 0.0
            pagerank_total = 0.0
            !$omp end single


            !$omp do schedule(static) reduction(+:diff) reduction(+:pagerank_total)
            DO source_i = 0, GRAPH_ORDER - 1
                new_pr = 0.0
                !$omp simd reduction(+:new_pr)
                do k = dest_offset(source_i), dest_offset(source_i+1)-1
                    destination_i = dest_map(k)
                    new_pr = new_pr + pagerank(destination_i) * outdeg_div_arr(destination_i)
                END DO
                new_pagerank(source_i) = DAMPING_FACTOR * new_pr + damping_value
                diff = diff + ABS(new_pagerank(source_i) - pagerank(source_i))
                pagerank_total = pagerank_total + new_pagerank(source_i)
            END DO


            !$omp do schedule(static)
            DO i = 0, GRAPH_ORDER - 1
                pagerank(i) = new_pagerank(i)
            END DO


            !$omp single
            max_diff = MAX(max_diff, diff)
            min_diff = MIN(min_diff, diff)
            total_diff = total_diff + diff

            IF (ABS(pagerank_total - 1.0_8) >= 1.0D-12) THEN
                WRITE(*, '(A,I0,A,F0.12)') '[ERROR] Iteration ', iteration, ': sum of all pageranks should be 1.0 (with a tolerance to 10-7 for floating-point inaccuracy) but the value observed is ', pagerank_total
            END IF

            iteration_end = omp_get_wtime()
            elapsed = omp_get_wtime() - start
            iteration = iteration + 1
            time_per_iteration = elapsed / iteration
            !$omp end single

        END DO
        !$omp end parallel
        WRITE(*, '(I0,A,F0.2,A)') iteration, ' iterations achieved in ', elapsed, ' seconds'
        RETURN
    END SUBROUTINE

    !> @brief Populates the edges in the graph for the challenge
    SUBROUTINE generate_sneaky_graph()
        IMPLICIT NONE

        INTEGER :: i
        INTEGER :: j
        INTEGER :: source
        INTEGER :: destination
        REAL(KIND=8) :: start

        WRITE(*, '(A)') 'Generate a graph for the challenge (i.e.: a sneaky graph :P )'
        start = omp_get_wtime()
        CALL initialize_graph()
        DO i = 0, GRAPH_ORDER - 1
            DO j = 0, GRAPH_ORDER - i - 1
                source = i
                destination = j
                IF (i .NE. j) THEN
                    adjacency_matrix(destination, source) = 1.0
                END IF
            END DO
        END DO
        WRITE(*, '(F0.2,A)') omp_get_wtime() - start, ' seconds to generate the graph.'
        RETURN
    END SUBROUTINE

    !> @brief Populates the edges in the graph for testing
    SUBROUTINE generate_nice_graph()
        IMPLICIT NONE

        INTEGER :: i
        INTEGER :: j
        INTEGER :: source
        INTEGER :: destination
        REAL(KIND=8) :: start

        WRITE(*, '(A)') 'Generate a graph for testing purposes (i.e.: a nice and conveniently designed graph :) ).'
        start = omp_get_wtime()
        CALL initialize_graph()
        DO i = 0, GRAPH_ORDER - 1
            DO j = 0, GRAPH_ORDER - 1
                source = i
                destination = j
                IF (i .NE. j) THEN
                    adjacency_matrix(destination, source) = 1.0
                END IF
            END DO
        END DO
        WRITE(*, '(F0.2,A)') omp_get_wtime() - start, ' seconds to generate the graph.'
        RETURN
        END
    END PROGRAM main
