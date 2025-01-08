// TESTAlgorithm.cpp: 实现文件
//

//#include "pch.h"
#include "stdafx.h"
#include "VSAlgorithmTask.h"
#include "afxdialogex.h"
#include "TESTAlgorithm.h"
#include "DllInterface.h"
#include "AIRuntimeDataStruct.h"
#include "AIRuntimeInterface.h"
#include "AIRuntimeUtils.h"
#include <math.h>

// TESTAlgorithm 对话框

IMPLEMENT_DYNAMIC(TESTAlgorithm, CDialogEx)

TESTAlgorithm::TESTAlgorithm(CWnd* pParent /*=nullptr*/)
	: CDialogEx(TESTAlgorithm::IDD, pParent)
{

}

TESTAlgorithm::~TESTAlgorithm()
{
    GetAIRuntime()->DestroyModle(0);
    GetAIRuntime()->DestoryRuntime();
}

void TESTAlgorithm::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
    DDX_Control(pDX, IDC_STATIC1, m_staticImage);
    DDX_Text(pDX, IDC_EDIT2, strFilePath);
    DDX_Text(pDX, IDC_EDIT3, listPath);
    DDX_Text(pDX, IDC_EDIT4, confPath);
    DDX_Text(pDX, IDC_EDIT9, onnxPath);
    DDX_Text(pDX, IDC_EDIT1, trtPath);
    DDX_Control(pDX, IDC_EDIT5, BatchSize);
}

BOOL TESTAlgorithm::OnInitDialog()
{
    CDialogEx::OnInitDialog();
    BatchSize.SetWindowText(_T("1"));

    return TRUE;
}
BEGIN_MESSAGE_MAP(TESTAlgorithm, CDialogEx)
	ON_BN_CLICKED(IDC_BUTTON1, &TESTAlgorithm::OnBnClickedButton1)
	ON_BN_CLICKED(IDC_BUTTON2, &TESTAlgorithm::OnBnClickedButton2)
    ON_BN_CLICKED(IDC_BUTTON3, &TESTAlgorithm::OnBnClickedButton3)
    ON_BN_CLICKED(IDC_BUTTON5, &TESTAlgorithm::OnBnClickedButton5)
    ON_EN_CHANGE(IDC_EDIT4, &TESTAlgorithm::OnEnChangeEdit4)

    ON_BN_CLICKED(IDC_BUTTON4, &TESTAlgorithm::OnBnClickedButton4)
    ON_EN_CHANGE(IDC_EDIT9, &TESTAlgorithm::OnEnChangeEdit9)
    ON_BN_CLICKED(IDC_BUTTON6, &TESTAlgorithm::OnBnClickedButton6)
    ON_STN_CLICKED(IDC_STATIC1, &TESTAlgorithm::OnStnClickedStatic1)
    ON_BN_CLICKED(IDC_BUTTON7, &TESTAlgorithm::OnBnClickedButton7)
    ON_BN_CLICKED(IDC_BUTTON8, &TESTAlgorithm::OnBnClickedButton8)
    ON_BN_CLICKED(IDC_BUTTON10, &TESTAlgorithm::OnBnClickedButton10)
    ON_BN_CLICKED(IDC_BUTTON9, &TESTAlgorithm::OnBnClickedButton9)
    ON_BN_CLICKED(IDC_BUTTON12, &TESTAlgorithm::OnBnClickedButton12)
    ON_BN_CLICKED(IDC_BUTTON11, &TESTAlgorithm::OnBnClickedButton11)
    ON_BN_CLICKED(IDC_BUTTON13, &TESTAlgorithm::OnBnClickedButton13)
END_MESSAGE_MAP()

// TESTAlgorithm 消息处理程序

void TESTAlgorithm::OnBnClickedButton2()
{
	// TODO: 在此添加控件通知处理程序代码
    // 创建 CFileDialog 对象
    CFileDialog ImgDlg(TRUE, NULL, NULL, OFN_FILEMUSTEXIST | OFN_HIDEREADONLY,
        _T("Image Files (*.bmp;*.jpg;*.jpeg;*.gif;*.png)|*.bmp;*.jpg;*.jpeg;*.gif;*.png||"), this);
        ImgDlg.DoModal();
        filePath = ImgDlg.GetPathName();
        theApp.WriteLog(eLOGTEST, eLOGLEVEL_SIMPLE, FALSE, TRUE, filePath.GetBuffer());
        GetDlgItem(IDC_EDIT2)->SetWindowText(filePath);
}

void TESTAlgorithm::OnBnClickedButton1()
{
    // TODO: 在此添加控件通知处理程序代码

}

void ParseParams(const CString& strParams, std::vector<CString>& vParams)
{
    vParams.clear();

    int nStart = 0, nEnd = 0;
    while (nEnd >= 0)
    {
        nEnd = strParams.Find(_T(","), nStart);
        CString strParam = nEnd >= 0 ? strParams.Mid(nStart, nEnd - nStart) : strParams.Right(strParams.GetLength() - nStart);
        strParam.Trim();
        if (!strParam.IsEmpty())
        {
            vParams.push_back(strParam);
        }
        nStart = nEnd + 1;
    }
}

void TESTAlgorithm::OnBnClickedButton3()
{
    // TODO: 在此添加控件通知处理程序代码

    //CFolderPickerDialog Folder(_T(""), 0, NULL, 0);
    CFileDialog Folder(TRUE);
    if (IDOK == Folder.DoModal())
    {
        strFilePath = Folder.GetPathName();
        UpdateData(FALSE);
    }
}

void TESTAlgorithm::OnBnClickedButton5()
{
    // TODO: 在此添加控件通知处理程序代码
    theApp.WriteLog(eLOGTEST, eLOGLEVEL_SIMPLE, FALSE, TRUE, _T("AI cls start. "));
    TaskInfoPtr spTaskInfo = std::make_shared<stTaskInfo>();
    // 设置任务参数
    spTaskInfo->inspParam = spTaskInfo;
    spTaskInfo->modelId = 0;
    std::string directoryPath = "F:\\dataset\\Lighting_ai\\0710\\G64\\ME0300\\G64_NG\\B61C320037B3BAL05_ME0300_4.jpg";
    //for (const auto& entry : fs::directory_iterator(directoryPath)) {
    //    if (entry.path().extension() == ".jpg") {
    //        std::cout << entry.path().string() << std::endl;

    //        spTaskInfo->imageData.emplace_back(cv::imread(entry.path().string()));
    //    }
    //}
    //cv::Mat src = cv::imread(directoryPath);
    //cv::imshow("sdas", src);
    //cv::waitKey();
    spTaskInfo->imageData.emplace_back(cv::imread((LPCSTR)(CStringA)strFilePath.GetBuffer()));
    eAIErrorCode commitTaskResult = GetAIRuntime()->CommitInferTask(spTaskInfo);
    theApp.WriteLog(eLOGTEST, eLOGLEVEL_SIMPLE, FALSE, TRUE, _T("AI cls end. "));
}

void TESTAlgorithm::OnEnChangeEdit4()
{
    // TODO:  如果该控件是 RICHEDIT 控件，它将不
    // 发送此通知，除非重写 CDialogEx::OnInitDialog()
    // 函数并调用 CRichEditCtrl().SetEventMask()，
    // 同时将 ENM_CHANGE 标志“或”运算到掩码中。

    // TODO:  在此添加控件通知处理程序代码
}

class ModelResultListener : public IModelResultListener {
public:
    void OnModelResult(ModelResultPtr spResult) override {
        // 打印收到的模型结果
        std::cout << "Received model result num : " << spResult->itemList.size() << std::endl;

        // 在这里对推理结果进行处理（例如打印结果）
        PrintInferenceResult(spResult->itemList);
    }

private:
    void PrintInferenceResult(std::vector<std::vector<stResultItem>> itemList) {
        // 打印推理结果
        for (auto i : itemList[0]) {
            //std::cout << "confidence = " << 1 / (1 + exp(-i.confidence)) << "\n"
            //    << "cls = " << LABEL[i.code] << std::endl;
            double conf = i.confidence;
            CString cstr(LABEL[i.code].c_str());
            theApp.WriteLog(eLOGTEST, eLOGLEVEL_SIMPLE, FALSE, TRUE, _T("AI Rturn Result. cls = %s,   confidence = %0.3f"), cstr, conf);
        }
    }
public:
    std::string LABEL[2] = { "ME0300&MU300_NG", "ME0300&MU300_OK" };
};
ModelResultListener resultListener;
//ModelResultListener resultListener1;
void TESTAlgorithm::OnBnClickedButton4()
{
    // TODO: 在此添加控件通知处理程序代码
    theApp.WriteLog(eLOGTEST, eLOGLEVEL_SIMPLE, FALSE, TRUE, _T("ResultListener start. "));
    //ModelResultListener resultListener;
    eAIErrorCode registerListenerResult = GetAIRuntime()->RegisterResultListener(0, &resultListener);
    //ModelResultListener resultListener1;
    //eAIErrorCode registerListenerResult1 = GetAIRuntime()->RegisterResultListener(1, &resultListener1);
    theApp.WriteLog(eLOGTEST, eLOGLEVEL_SIMPLE, FALSE, TRUE, _T("ResultListener end. "));
    //const cv::String cvStr = CT2A(filePath.GetBuffer());
    //cv::Mat matSrcBuf = imread(cvStr, IMREAD_UNCHANGED);
    //double* dPara = theApp.GetAlignParameter(0);
    //for (int i = 0; i < 20; i++) {
    //    double d = *(dPara + i);
    //    theApp.WriteLog(eLOGTEST, eLOGLEVEL_SIMPLE, FALSE, TRUE, _T("%d = %.2f"), i, d);
    //}
    //double theta;
    //cv::Point ptCellCenter;
    //int res = Align_FindTheta(matSrcBuf, dPara, theta, ptCellCenter, filePath.GetBuffer());
    //theApp.WriteLog(eLOGTEST, eLOGLEVEL_SIMPLE, FALSE, TRUE,
    //    _T("Success Seq_StartAlign. PanelID : %s, Theta : %lf, Cell Center X : %d, Y : %d"),
    //    filePath.GetBuffer(), theta, ptCellCenter.x, ptCellCenter.y);
    //CString str;
    //str.Format(_T("%.2f"), theta);
    //GetDlgItem(IDC_EDIT8)->SetWindowText((LPCTSTR)str);
    //str.Format(_T("[%.2f,%.2f,%.2f,%.2f]"), ptCellCenter.x, ptCellCenter.y, ptCellCenter.x, ptCellCenter.y);
    //GetDlgItem(IDC_EDIT7)->SetWindowText((LPCTSTR)str);
}

void TESTAlgorithm::OnEnChangeEdit9()
{
    // TODO:  如果该控件是 RICHEDIT 控件，它将不
    // 发送此通知，除非重写 CDialogEx::OnInitDialog()
    // 函数并调用 CRichEditCtrl().SetEventMask()，
    // 同时将 ENM_CHANGE 标志“或”运算到掩码中。

    // TODO:  在此添加控件通知处理程序代码
}

void TESTAlgorithm::OnBnClickedButton6()
{
    // TODO: 在此添加控件通知处理程序代码
    theApp.WriteLog(eLOGTEST, eLOGLEVEL_SIMPLE, FALSE, TRUE, _T("Model Init start. "));
    std::string fconfig = "F:\\code\\AIFrameworkTEST\\config.json";
    //AI_Initialization(fconfig);fconfig.c_str()
    //auto config = read_json_from_file((CStringA)confPath.GetBuffer());
    std::ifstream i((CStringA)confPath.GetBuffer());
    //std::ifstream testjs;
    //testjs()
    json config;
    i >> config;
    // Initilize the AI Runtime
    stAIConfigInfo aiIniConfig(config["initConfig"]);
    GetAIRuntime()->InitRuntime(aiIniConfig);

    // Initilize AI models.
    json temp = config["modelInfo"];
    GetAIRuntime()->CreateModle(temp);
    theApp.WriteLog(eLOGTEST, eLOGLEVEL_SIMPLE, FALSE, TRUE, _T("Model Init end. "));
}

void TESTAlgorithm::OnStnClickedStatic1()
{
    // TODO: 在此添加控件通知处理程序代码
}

void TraverseDirectory(const CString& path)
{
    CFileFind finder;
    CString searchPath = path + _T("\\*.*");

    BOOL bWorking = finder.FindFile(searchPath);
    while (bWorking)
    {
        bWorking = finder.FindNextFile();

        if (finder.IsDots())
        {
            continue;
        }

        if (finder.IsDirectory())
        {
            CString subDir = finder.GetFilePath();
            // 处理子目录
            TraverseDirectory(subDir);
        }
        else
        {
            CString filePath = finder.GetFilePath();
            // 处理文件
            AfxMessageBox(filePath);
        }
    }
}
void TESTAlgorithm::OnBnClickedButton7()
{
    // TODO: 在此添加控件通知处理程序代码
    CFolderPickerDialog Folder(_T(""), 0, NULL, 0);
    if (IDOK == Folder.DoModal())
    {
        listPath = Folder.GetPathName();
        UpdateData(FALSE);
    }
    //TraverseDirectory(listPath);

}

void TESTAlgorithm::OnBnClickedButton8()
{
    // TODO: 在此添加控件通知处理程序代码
    //CFolderPickerDialog Folder(_T(""), 0, NULL, 0);
    CFileDialog Folder(TRUE);
    if (IDOK == Folder.DoModal())
    {
        confPath = Folder.GetPathName();
        UpdateData(FALSE);
    }
}

void TESTAlgorithm::OnBnClickedButton10()
{
    // TODO: 在此添加控件通知处理程序代码
    theApp.WriteLog(eLOGTEST, eLOGLEVEL_SIMPLE, FALSE, TRUE, _T("ResultListener start. "));
    //ModelResultListener resultListener;
    eAIErrorCode registerListenerResult = GetAIRuntime()->UnregisterResultListener(&resultListener);
    //ModelResultListener resultListener1;
    //eAIErrorCode registerListenerResult1 = GetAIRuntime()->RegisterResultListener(1, &resultListener1);
    theApp.WriteLog(eLOGTEST, eLOGLEVEL_SIMPLE, FALSE, TRUE, _T("ResultListener end. "));
}

void TESTAlgorithm::OnBnClickedButton9()
{
    // TODO: 在此添加控件通知处理程序代码
    CFileFind finder;
    CString searchPath = listPath + _T("\\*.*");

    BOOL bWorking = finder.FindFile(searchPath);
    while (bWorking)
    {
        bWorking = finder.FindNextFile();

        if (finder.IsDots())
        {
            continue;
        }

        if (finder.IsDirectory())
        {
            CString subDir = finder.GetFilePath();
            // 处理子目录
            TraverseDirectory(subDir);
        }
        else
        {
            CString filePath = finder.GetFilePath();
            // 处理文件
            //AfxMessageBox(filePath);
            theApp.WriteLog(eLOGTEST, eLOGLEVEL_SIMPLE, FALSE, TRUE, _T("AI cls start. "));
            TaskInfoPtr spTaskInfo = std::make_shared<stTaskInfo>();
            // 设置任务参数
            spTaskInfo->inspParam = spTaskInfo;
            spTaskInfo->modelId = 0;
            spTaskInfo->imageData.emplace_back(cv::imread((LPCSTR)(CStringA)filePath.GetBuffer()));
            eAIErrorCode commitTaskResult = GetAIRuntime()->CommitInferTask(spTaskInfo);
            theApp.WriteLog(eLOGTEST, eLOGLEVEL_SIMPLE, FALSE, TRUE, _T("AI cls end. "));
        }
    }
}

void TESTAlgorithm::OnBnClickedButton12()
{
    // TODO: 在此添加控件通知处理程序代码
    CFileDialog Folder(TRUE);
    if (IDOK == Folder.DoModal())
    {
        onnxPath = Folder.GetPathName();
        UpdateData(FALSE);
    }
}

CString ExtractFileName(const CString& onnxPath) {
    // 查找最后一个反斜杠的位置
    int lastIndex = onnxPath.ReverseFind('\\');

    // 提取反斜杠之后的部分
    CString fileName = onnxPath.Mid(lastIndex + 1);

    // 去除扩展名
    int dotIndex = fileName.ReverseFind('.');
    if (dotIndex != -1) {
        fileName = fileName.Left(dotIndex);
    }

    // 返回文件名
    return fileName;
}
void TESTAlgorithm::OnBnClickedButton11()
{
    // TODO: 在此添加控件通知处理程序代码
    // 
    BatchSize.EnableWindow(FALSE);
    LPWSTR lpwstronnx = onnxPath.GetBuffer();
    int onnxsize = WideCharToMultiByte(CP_UTF8, 0, lpwstronnx, -1, NULL, 0, NULL, NULL);
    char* onnxcstr = new char[onnxsize];
    WideCharToMultiByte(CP_UTF8, 0, lpwstronnx, -1, onnxcstr, onnxsize, NULL, NULL);
    theApp.WriteLog(eLOGTEST, eLOGLEVEL_SIMPLE, FALSE, TRUE, _T("ONNX to TRT Model Start. "));
    CString fileName = ExtractFileName(onnxPath);
    CString trtname;
    trtname.Format(_T("\\%s.trtmodel"), fileName);
    trtPath.Append(trtname);
    LPWSTR lpwstrtrt = trtPath.GetBuffer();
    int trtsize = WideCharToMultiByte(CP_UTF8, 0, lpwstrtrt, -1, NULL, 0, NULL, NULL);
    char* trtcstr = new char[trtsize];
    WideCharToMultiByte(CP_UTF8, 0, lpwstrtrt, -1, trtcstr, trtsize, NULL, NULL);
    // 
    CString strValue;
    BatchSize.GetWindowText(strValue);
    int nValue = _ttoi(strValue);
    // model type: 0 --> TRT::FP32   1--> TRT::FP16
    BOOL status = build_model(
        1,
        nValue,
        onnxcstr,
        trtcstr
    );
    BatchSize.EnableWindow(TRUE);
    theApp.WriteLog(eLOGTEST, eLOGLEVEL_SIMPLE, FALSE, TRUE, _T("ONNX to TRT Model End. status = %d  model_type = %d  max_batch_size = %d"), int(status), 1, nValue);
    onnxPath.ReleaseBuffer();
    trtPath.ReleaseBuffer();
}

void TESTAlgorithm::OnBnClickedButton13()
{
    // TODO: 在此添加控件通知处理程序代码

    CFolderPickerDialog Folder(_T(""), 0, NULL, 0);
    if (IDOK == Folder.DoModal())
    {
        trtPath = Folder.GetPathName();
        UpdateData(FALSE);
    }

}

void TESTAlgorithm::OnEnChangeEdit1()
{
    // TODO:  如果该控件是 RICHEDIT 控件，它将不
    // 发送此通知，除非重写 CDialogEx::OnInitDialog()
    // 函数并调用 CRichEditCtrl().SetEventMask()，
    // 同时将 ENM_CHANGE 标志“或”运算到掩码中。

    // TODO:  在此添加控件通知处理程序代码
}
