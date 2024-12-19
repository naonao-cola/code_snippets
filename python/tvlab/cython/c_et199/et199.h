#ifndef _FEITIAN_ET199_32_H_
#define _FEITIAN_ET199_32_H_

  

#define   HANDLE  void *;

#define   ET_S_SUCCESS			0x00000000	// 操作成功
#define   ET_E_KEY_REMOVED		0xF0000001	// 设备未连接，或者被移除
#define   ET_E_INVALID_PARAMETER	0xF0000002	// 参数错误
#define   ET_E_COMM_ERROR		0xF0000003	// 通讯错误，例如数据传输超时
#define   ET_E_INSUFFICIENT_BUFFER	0xF0000004	// 缓冲区空间不足
#define   ET_E_NO_LIST			0xF0000005	// 没有找到设备列表
#define   ET_E_DEVPIN_NOT_CHECK		0xF0000006	// 开发商口令没有验证
#define   ET_E_USERPIN_NOT_CHECK	0xF0000007	// 用户口令没有验证
#define	  ET_E_RSA_FILE_FORMAT_ERROR	0xF0000008	// RSA文件格式错误
#define   ET_E_DIR_NOT_FOUND		0xF0000009	// 目录没有找到
#define   ET_E_ACCESS_DENIED		0xF000000A	// 访问被拒绝
#define   ET_E_ALREADY_INITIALIZED	0xF000000B	// 产品已经初始化
#define   ET_E_INCORRECT_PIN		0xF0000C00	// 密码不正确
#define   ET_E_DF_SIZE			0xF000000D	// 指定的目录空间大小不够
#define   ET_E_FILE_EXIST		0xF000000E	// 文件已存在
#define   ET_E_UNSUPPORTED		0xF000000F	// 功能不支持或尚未建立文件系统
#define   ET_E_FILE_NOT_FOUND		0xF0000010	// 未找到指定的文件
#define	  ET_E_ALREADY_OPENED		0xF0000011	// 卡已经被打开
#define   ET_E_DIRECTORY_EXIST		0xF0000012	// 目录已存在
#define   ET_E_CODE_RANGE		0xF0000013	// 虚拟机内存地址溢出
#define   ET_E_INVALID_POINTER		0xF0000014	// 虚拟机错误的指针 
#define   ET_E_GENERAL_FILESYSTEM	0xF0000015	// 常规文件系统错误 
#define   ET_E_OFFSET_BEYOND		0xF0000016	// 文件偏移量超出文件大小
#define   ET_E_FILE_TYPE_MISMATCH	0xF0000017	// 文件类型不匹配
#define   ET_E_PIN_BLOCKED		0xF0000018	// PIN码锁死
#define   ET_E_INVALID_CONTEXT		0xF0000019	// ETContext 参数错误
#define   ET_E_SHARING_VIOLATION	0XF000001A	// 另一个程序正在使用此文件，进程无法访问
//linux下新增错误码
#define   ET_TOOMUCHTHREAD		0XF000001B     //同一个进程中打开锁的线程数 > 100

#define   ET_E_ERROR_UNKNOWN		0xFFFFFFFF	// 未知的错误
#define   ET_E_LOAD_FILE_FAILED		0xF0001001

#define	  MAX_ATR_LEN	16	// 最大ATR长度 
#define   MAX_ID_LEN	8       // 最大硬件ID长度

#define   ET_USER_PIN	0x00000000	// 用户PIN
#define   ET_DEV_PIN	0x00000001      // 开发商PIN

#define   ET_CREATE_NEW		0x00000000	// 创建新文件
#define   ET_UPDATE_FILE	0x00000001      // 更新数据文件

#define   ET_CREATE_ROOT_DIR	0x00000000      // 创建根目录
#define   ET_CREATE_SUB_DIR	0x00000001      // 创建当前目录的子目录


#define   ET_LED_UP	0x00000001	// LED灯亮
#define   ET_LED_DOWN	0x00000002	// LED灯灭
#define   ET_LED_WINK 	0x00000003	// LED灯闪烁


#define   ET_GET_DEVICE_TYPE		0x00000011	// 获得设备类型
#define   ET_GET_SERIAL_NUMBER		0x00000012      // 获取硬件序列号
#define   ET_GET_DEVICE_USABLE_SPACE	0x00000013      // 获得设备空间大小
#define   ET_GET_DEVICE_ATR		0x00000014      // 获得设备ATR
#define   ET_GET_CUSTOMER_NAME		0x00000015      // 获得客户号
#define   ET_GET_MANUFACTURE_DATE	0x00000016      // 获得生产日期
#define   ET_GET_DF_AVAILABLE_SPACE	0x00000017      // 获得当前目录的剩余空间
#define   ET_GET_EF_INFO		0x00000018      // 获得指定文件信息
#define   ET_GET_COS_VERSION		0x00000019	// 获得COS版本信息

#define   ET_SET_DEVICE_ATR		0x00000021      // 设置设备ATR
#define   ET_SET_DEVICE_TYPE		0x00000022	// 设置设备类型
#define   ET_SET_SHELL_KEY		0x00000023	// 设置8字节外壳加密种子码
#define	  ET_SET_CUSTOMER_NAME		0x00000024	// 输入一个种子, 产生客户号

#define   ET_RESET_DEVICE		0x00000031      // 复位设备

#define ET_GET_CURRENT_TIME	0x00000032	/* get current time */
#define ET_GET_PRODUCT_NAMEID	0x00000033	/* get product name ID */



#define ET_DEVICE_TYPE_PKI	0x01	// 身份验证锁类型
#define ET_DEVICE_TYPE_DONGLE	0x02	// 加密锁类型
#define ET_DEVICE_TYPE_EMPTY	0x04	// 空锁类型


#define   ET_DEFAULT_TRY	0xFF	// 默认重试次数（无限次）

#define	  ET_DEFAULT_DEV_PIN	(unsigned char *)"123456781234567812345678"
#define   ET_DEFAULT_USER_PIN	(unsigned char *)"12345678"
#define   ET_DEV_PIN_LEN	24
#define   ET_USER_PIN_LEN	8


#define	 ET_EXCLUSIVE_MODE	0        // 独占模式
#define  ET_SHARE_MODE		1        // 共享模式

#ifndef	 ET199_FILE_TYPE
#define  ET199_FILE_TYPE

    
#define  FILE_TYPE_EXE		0	// 可执行文件
#define  FILE_TYPE_DATA		1	// 数据文件
#define  FILE_TYPE_RSA_PUBLIC	2	// RSA 公钥文件
#define  FILE_TYPE_RSA_PRIVATE	3	// RSA 私钥文件
#define  FILE_TYPE_INTERNAL_EXE	4	// 可执行文件（不可读写）
   
#endif

struct  ET_CONTEXT {
        int dwIndex;			// 以0开始的设备索引
        unsigned long dwVersion;	// 设备COS版本
        //HANDLE hLock;			// 设备句柄
        unsigned char reserve[12];	// 保留
        unsigned long dwCustomer;	// 客户号
        unsigned char bAtr[MAX_ATR_LEN];// ATR
        unsigned char bID[MAX_ID_LEN];	// 硬件ID
        unsigned long dwAtrLen;		// ATR长度
};

struct  ET_MANUFACTURE_DATE {
        unsigned char byYear;		// 年
        unsigned char byMonth;		// 月
        unsigned char byDay;		// 日
	unsigned char byHour;		// 时
        unsigned char byMinute;		// 分
        unsigned char bySecond;		// 秒
};


    #ifndef ET199_FILEINFO
    #define ET199_FILEINFO
    #pragma pack (1)
    struct  EFINFO
	{
        unsigned short wFileID;																// 文件ID
        unsigned char bFileType;															// 文件类型
        unsigned short wFileSize;															// 文件大小
    };
    #pragma pack ()
    #endif


    struct ET_CREATEDIRINFO
	{
        unsigned long dwCreateDirInfoSize;													// 结构体大小
        unsigned char  szAtr[MAX_ATR_LEN];													// ATR 字符串
    };

	struct ET_OPENINFO 
	{
		unsigned long dwOpenInfoSize;														// 结构体大小
		unsigned long dwShareMode;															// 共享模式
	};

	//====================ET199 COS RSA TLV==================================================
	#define STRUCT_COS_TLV_(Tlv_Length)									   \
	struct																   \
	{																	   \
		unsigned char bTagHigh;											   \
		unsigned char bTagLow;											   \
		unsigned char bLenHigh;											   \
		unsigned char bLenLow;											   \
		unsigned char bData[Tlv_Length];								   \
	}
	//===============ET199 COS RSA 公钥======================================================

	#define STRUCT_COS_RSA_PUBLIC_KEY_(Bits)								\
	struct																	\
	{																		\
		STRUCT_COS_TLV_(Bits/8)		n;										\
		STRUCT_COS_TLV_(4)			e;										\
	}

	typedef STRUCT_COS_RSA_PUBLIC_KEY_(512 )	COS_RSA_PUBLIC_KEY_512;
	typedef STRUCT_COS_RSA_PUBLIC_KEY_(1024)	COS_RSA_PUBLIC_KEY_1024;
	typedef STRUCT_COS_RSA_PUBLIC_KEY_(2048)	COS_RSA_PUBLIC_KEY_2048;

	//===============ET199 COS RSA 私钥======================================================
	#define STRUCT_COS_RSA_CRT_PRIVATE_KEY_(Bits)							\
	struct																	\
	{																		\
		STRUCT_COS_TLV_(Bits/16)	p;										\
		STRUCT_COS_TLV_(Bits/16)	q;										\
		STRUCT_COS_TLV_(Bits/16)	dp;										\
		STRUCT_COS_TLV_(Bits/16)	dq;										\
		STRUCT_COS_TLV_(Bits/16)	InvQ;									\
	}

	typedef STRUCT_COS_RSA_CRT_PRIVATE_KEY_(512 )	COS_RSA_CRT_PRIVATE_KEY_512;
	typedef STRUCT_COS_RSA_CRT_PRIVATE_KEY_(1024)	COS_RSA_CRT_PRIVATE_KEY_1024;
	typedef STRUCT_COS_RSA_CRT_PRIVATE_KEY_(2048)	COS_RSA_CRT_PRIVATE_KEY_2048;

	#define STRUCT_COS_RSA_PRIVATE_KEY_(Bits)								\
	struct																	\
	{																		\
		STRUCT_COS_TLV_(Bits/8)	n;										    \
		STRUCT_COS_TLV_(Bits/8)	d;										    \
	}

	typedef STRUCT_COS_RSA_PRIVATE_KEY_(512 )	COS_RSA_PRIVATE_KEY_512;
	typedef STRUCT_COS_RSA_PRIVATE_KEY_(1024)	COS_RSA_PRIVATE_KEY_1024;
	typedef STRUCT_COS_RSA_PRIVATE_KEY_(2048)	COS_RSA_PRIVATE_KEY_2048;
    
//===================函数定义===================================


//
unsigned long  ETEnum(struct ET_CONTEXT *pETContextList, int * pdwDeviceCount);
unsigned long  ETOpen(struct ET_CONTEXT *pETCtx);
unsigned long  ETClose(struct ET_CONTEXT * pETCtx);
unsigned long  ETControl(struct ET_CONTEXT *pETCtx, unsigned int dwCtlCode, unsigned char *pInBuffer, unsigned long dwInBufferLen, unsigned char *pOutBuffer, unsigned int dwOutBufferLen, unsigned long *pdwBytesReturned);
unsigned long  ETCreateDir(struct ET_CONTEXT * pETCtx,  char* lpszDirID,  unsigned long dwDirSize,  unsigned long dwFlags);
unsigned long  ETCreateDirEx(struct ET_CONTEXT * pETCtx, char* lpszDirID,  unsigned long dwDirSize,  unsigned long dwFlags,  struct ET_CREATEDIRINFO * pCreateDirInfo);
unsigned long  ETChangeDir(struct ET_CONTEXT * pETCtx,  char* lpszPath);
unsigned long  ETEraseDir(struct ET_CONTEXT	* pETCtx,  char* lpszDirID);
unsigned long  ETVerifyPin(struct ET_CONTEXT * pETCtx,  unsigned char * pbPin,  unsigned long dwPinLen,  unsigned long dwPinType);
unsigned long  ETChangePin(struct ET_CONTEXT * pETCtx,  unsigned char * pbOldPin,  unsigned long dwOldPinLen,  unsigned char * pbNewPin,  unsigned long dwNewPinLen, unsigned long dwPinType, unsigned char byPinTryCount);
unsigned long  ETCreateFile(struct ET_CONTEXT * pETCtx,  char* lpszFileID,  unsigned long dwFileSize,  unsigned char bFileType);
unsigned long  ETWriteFile(struct ET_CONTEXT * pETCtx,  char* lpszFileID,  unsigned long dwOffset,  unsigned char * pBuffer,  unsigned long dwBufferSize);
unsigned long  ETWriteFileEx(struct ET_CONTEXT	*pETCtx,  char* lpszFileID,  unsigned long	dwOffset, unsigned char * pBuffer,  unsigned long	dwBufferSize, unsigned long dwFileSize, unsigned long * pdwBytesWritten, unsigned long dwFlags, unsigned char bFileType);
unsigned long  ETExecute(struct ET_CONTEXT * pETCtx,  char* lpszFileID,  unsigned char * pInBuffer, unsigned long dwInbufferSize, unsigned char * pOutBuffer, unsigned long dwOutBufferSize, unsigned long * pdwBytesReturned);
unsigned long  PETWriteFile (struct ET_CONTEXT * pETCtx, char* lpszFileID, char* lpszPCFilePath, unsigned long * pdwFileSize, unsigned long dwFlags, unsigned char bFileType, unsigned long *pdwBytesWritten);
unsigned long  ETGenRsaKey(struct ET_CONTEXT *pETCtx,  unsigned short wKeySize,  unsigned long dwE,  char*	lpszPubFileID,  char* lpszPriFileID, unsigned char * pbPubKeyData, unsigned long * dwPubKeyDataSize, unsigned char * pbPriKeyData, unsigned long * dwPriKeyDataSize);
unsigned long  ETFormatErrorMessage( unsigned long dwRet, char*	lpszMessage,  unsigned long dwMsgBufLen);

//==============================================================

#endif

