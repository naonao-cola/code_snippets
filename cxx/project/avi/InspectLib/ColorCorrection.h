

/************************************************************************/
//SVI色彩校正相关标头
//修改日期:18.02.20
/************************************************************************/

#pragma once

#include "Define.h"

//颜色校正数量
#define COLOR_CORRECTION_COUNT	9

class CColorCorrection
{
public:
	CColorCorrection(void);
	virtual ~CColorCorrection(void);

public:
		//Color校正值Load
	long	ColorCorrectionLoad(CString strFileName);

		//导入颜色校正值
	double*	GetColorCorrection();

protected:
		//确认是否加载了
	bool	bLoad;

		//色彩校正值
	double	m_dCoefficient[COLOR_CORRECTION_COUNT];

};