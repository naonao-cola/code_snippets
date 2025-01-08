#pragma once


#define TE_OK                     0

#define TE_NOT_READY              10001
#define TE_TIME_OUT               10002
#define TE_WRONG_PARAM            10003
#define TE_DUPLICATE_NAME         10004
#define TE_INVALID_IMG_DATA       10005
#define TE_QUEUE_OVERFLOW         10006
#define TE_WRONG_STATE            10007
#define TE_CREATE_DIRECTORY_FAIL  10008
#define TE_KEY_NOT_FOUND          10009
#define TE_CLASS_NOT_FOUND        10010
#define TE_NOT_IMPLEMENT          10011
#define TE_LOAD_MODEL_FAIL        10012
#define TE_SEARCH_NO_RESULT       10013


#define FAIL_RETURN(err) if (err != TE_OK) return err;



