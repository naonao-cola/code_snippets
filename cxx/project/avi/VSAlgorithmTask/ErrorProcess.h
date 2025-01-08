#pragma once

// Inaf DLL Exception Handler
#ifndef _SEQUENCE_TASK_EXCEPTION_PROCESS_H 
#define _SEQUENCE_TASK_EXCEPTION_PROCESS_H

#include <windows.h>

/************************************************************************/
/*					System Exception Error                              */
/************************************************************************/
#define SEQUENCE_TASK_EXCEPTION_ACCESS_VIOLATION			1000	
#define SEQUENCE_TASK_EXCEPTION_BREAKPOINT					1001
#define SEQUENCE_TASK_EXCEPTION_DATATYPE_MISALIGNMENT		1002
#define SEQUENCE_TASK_EXCEPTION_SINGLE_STEP					1003
#define SEQUENCE_TASK_EXCEPTION_ARRAY_BOUNDS_EXCEEDED		1004
#define SEQUENCE_TASK_EXCEPTION_FLT_DENORMAL_OPERAND		1005
#define SEQUENCE_TASK_EXCEPTION_FLT_DIVIDE_BY_ZERO			1006
#define SEQUENCE_TASK_EXCEPTION_FLT_INEXACT_RESULT			1007
#define SEQUENCE_TASK_EXCEPTION_FLT_INVALID_OPERATION		1008
#define SEQUENCE_TASK_EXCEPTION_FLT_OVERFLOW				1009
#define SEQUENCE_TASK_EXCEPTION_FLT_STACK_CHECK				1010
#define SEQUENCE_TASK_EXCEPTION_FLT_UNDERFLOW				1011
#define SEQUENCE_TASK_EXCEPTION_INT_DIVIDE_BY_ZERO			1012
#define SEQUENCE_TASK_EXCEPTION_INT_OVERFLOW				1013
#define SEQUENCE_TASK_EXCEPTION_PRIV_INSTRUCTION			1014
#define SEQUENCE_TASK_EXCEPTION_NONCONTINUABLE_EXCEPTION	1015

#define SEQUENCE_TASK_EXCEPTION_UNKNOWN_EXCEPTION			1100

/************************************************************************/
/*							Error Code                                  */
/************************************************************************/

#define SEQUENCE_TASK_PARAMETER_INVALID_ERROR				400		// Parameter Invalid Error 

#define SEQUENCE_TASK_APPLICATION_ERROR						1200

#define SEQUENCE_TASK_SEQUENCE_IS_NOTHING					1500
#define SEQUENCE_TASK_SEQUENCE_IS_BUSY						1501
#define SEQUENCE_TASK_SEQUENCE_IS_NOT_BUSY					1502
#define SEQUENCE_TASK_SEQUENCE_IS_NOT_EXECUTABLE			1505		// Indexer Initial All　中

#define SEQUENCE_TASK_SEQUENCE_PARAMETER_IS_WRONG			1511
#define SEQUENCE_TASK_SEQUENCE_FAIL							1512

/************************************************************************/
/*							Error 処理 MACRO							*/
/************************************************************************/
// C Style
#define START_EXCEPTION		int  uExceptCode; _try 
#define	END_EXCEPTION		__except(uExceptCode = SEQUENCE_TASK_Exception_Process(GetExceptionCode())){nRet = uExceptCode;}
// C++ Style 
#define EXCEPTION_TRY	try{
#define EXCEPTION_CATCH	}catch(int nErrCode){nRet = nErrCode;}	catch(...){nRet = SEQUENCE_TASK_EXCEPTION_UNKNOWN_EXCEPTION;}

// Exception Error処理関数。
// 実現理由はSystem Error CodeはLONG形だが,ASALM基本関数のReturn形はUSHORTだから,Error Codeの変更のために。
__inline int	SEQUENCE_TASK_Exception_Process(long ErrorCode)
{
	int nErrorCode;

		//必要なら下のCase 内にError処理する。
	switch ( ErrorCode )
	{
	case EXCEPTION_ACCESS_VIOLATION:
		nErrorCode = SEQUENCE_TASK_EXCEPTION_ACCESS_VIOLATION;
		break;
	case EXCEPTION_BREAKPOINT:
		nErrorCode = SEQUENCE_TASK_EXCEPTION_BREAKPOINT;
		break;
	case EXCEPTION_DATATYPE_MISALIGNMENT:
		nErrorCode = SEQUENCE_TASK_EXCEPTION_DATATYPE_MISALIGNMENT;
		break;
	case EXCEPTION_SINGLE_STEP:
		nErrorCode = SEQUENCE_TASK_EXCEPTION_SINGLE_STEP;
		break;
	case EXCEPTION_ARRAY_BOUNDS_EXCEEDED:
		nErrorCode = SEQUENCE_TASK_EXCEPTION_ARRAY_BOUNDS_EXCEEDED;
		break;
	case EXCEPTION_FLT_DENORMAL_OPERAND:
		nErrorCode = SEQUENCE_TASK_EXCEPTION_FLT_DENORMAL_OPERAND;
		break;
	case EXCEPTION_FLT_DIVIDE_BY_ZERO:
		nErrorCode = SEQUENCE_TASK_EXCEPTION_FLT_DIVIDE_BY_ZERO;
		break;
	case EXCEPTION_FLT_INEXACT_RESULT:
		nErrorCode = SEQUENCE_TASK_EXCEPTION_FLT_INEXACT_RESULT;
		break;
	case EXCEPTION_FLT_INVALID_OPERATION:
		nErrorCode = SEQUENCE_TASK_EXCEPTION_FLT_INVALID_OPERATION;
		break;
	case EXCEPTION_FLT_OVERFLOW:
		nErrorCode = SEQUENCE_TASK_EXCEPTION_FLT_OVERFLOW;
		break;
	case EXCEPTION_FLT_STACK_CHECK:
		nErrorCode = SEQUENCE_TASK_EXCEPTION_FLT_STACK_CHECK;
		break;
	case EXCEPTION_FLT_UNDERFLOW:
		nErrorCode = SEQUENCE_TASK_EXCEPTION_FLT_UNDERFLOW;
		break;
	case EXCEPTION_INT_DIVIDE_BY_ZERO:
		nErrorCode = SEQUENCE_TASK_EXCEPTION_INT_DIVIDE_BY_ZERO;
		break;
	case EXCEPTION_INT_OVERFLOW:
		nErrorCode = SEQUENCE_TASK_EXCEPTION_INT_OVERFLOW;
		break;
	case EXCEPTION_PRIV_INSTRUCTION:
		nErrorCode = SEQUENCE_TASK_EXCEPTION_PRIV_INSTRUCTION;
		break;
	case EXCEPTION_NONCONTINUABLE_EXCEPTION:
		nErrorCode = SEQUENCE_TASK_EXCEPTION_NONCONTINUABLE_EXCEPTION;
		break;
	default:
		nErrorCode = SEQUENCE_TASK_EXCEPTION_UNKNOWN_EXCEPTION;
		break;
	}

	return nErrorCode;
}

#endif	//_SEQUENCE_TASK_EXCEPTION_PROCESS_H