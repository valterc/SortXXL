#ifndef __HANDLE_ERROR_H__
#define __HANDLE_ERROR_H__
/*-------------------------------------------------------------------
 * Function to process CUDA errors
 * @param err [IN] CUDA error to process (usually the code returned
 *	 by the cuda function)
 * @param line [IN] line of source code where function is called
 * @param file [IN] name of source file where function is called
 * @return on error, the function terminates the process with 
 * 		EXIT_FAILURE code.
 * source: "CUDA by Example: An Introduction to General-Purpose "
 * GPU Programming", Jason Sanders, Edward Kandrot, NVIDIA, July 2010
 * @note: the function should be called through the 
 * 	macro 'HANDLE_ERROR'
 *------------------------------------------------------------------*/
static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) 
    {
        printf( "[ERROR] '%s' (%d) in '%s' at line '%d'\n",
		cudaGetErrorString(err),err,file,line);
        exit( EXIT_FAILURE );
    }
}
/* The HANDLE_ERROR macro */
#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__ ))

/* 
 * Function that check a kernel execution, exiting 
 * if execution errors are found. The function should not be 
 * called directly, it should be called through the CHECK_KERNEL_EXEC
 * macro.
 */
static void CheckKernelExec(const char *file,const int line){
        cudaError_t err;
        err = cudaGetLastError();
        if( err != cudaSuccess ){
                fprintf(stderr,"[ERROR] '%s' (%d) in '%s' at line '%d'\n",
                                cudaGetErrorString(err),err,file,line);
                exit( EXIT_FAILURE );
        }
}

#define CHECK_KERNEL_EXEC()     CheckKernelExec(__FILE__,__LINE__)

#endif
