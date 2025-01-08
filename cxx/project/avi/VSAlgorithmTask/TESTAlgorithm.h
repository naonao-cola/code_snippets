#pragma once
#include "afxdialogex.h"

// TESTAlgorithm 对话框

class TESTAlgorithm : public CDialogEx
{
	DECLARE_DYNAMIC(TESTAlgorithm)

public:
	TESTAlgorithm(CWnd* pParent = nullptr);   // 标准构造函数
	virtual ~TESTAlgorithm();

// 对话框数据
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_DIALOG1 };
#endif
	enum { IDD = IDD_DIALOG1 };
protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 支持
	virtual BOOL OnInitDialog();

	DECLARE_MESSAGE_MAP()
public:
	afx_msg void OnBnClickedButton1();
	afx_msg void OnBnClickedButton2();

public:
	CString filePath;
	afx_msg void OnBnClickedButton3();
	afx_msg void OnBnClickedButton5();
	afx_msg void OnEnChangeEdit4();
	afx_msg void OnEnChangeEdit6();
	afx_msg void OnLbnSelchangeList3();
	afx_msg void OnBnClickedButton4();
	afx_msg void OnEnChangeEdit9();
public:

	afx_msg void OnBnClickedButton6();
	afx_msg void OnStnClickedStatic1();
	CStatic m_staticImage;
	CBrush m_blackBrush;

	std::string img_path;
	CString strFilePath;
	CString listPath;
	CString confPath;
	CString onnxPath;
	CString trtPath;
	CEdit BatchSize;
	afx_msg void OnBnClickedButton7();
	afx_msg void OnBnClickedButton8();
	afx_msg void OnBnClickedButton10();
	afx_msg void OnBnClickedButton9();
	afx_msg void OnBnClickedButton12();
	afx_msg void OnBnClickedButton11();
	afx_msg void OnBnClickedButton13();
	afx_msg void OnEnChangeEdit1();
};

