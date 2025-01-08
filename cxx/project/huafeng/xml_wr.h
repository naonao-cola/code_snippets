/**
 * @file    xml_wr.h
 * @author  ����
 * @brief   xml��д
 * @details Ϊ������ģ�Ϊ����������Ϊ��ʥ�̾�ѧ��Ϊ������̫ƽ��
 * @version 1.0.0.1
 * @date    2021��1��4��15:27:22
 * @copyright Copyright (c) 2050
 *
 **********************************************************************************
 * @par �޸���־:
 * <table>
 * <tr><th>Date                 <th>Version    <th>Author     <th>Description
 * <tr><td> 2021��1��7��13:04:43<td>1.0.0.2    <td>naonao     <td>����д�ֿ�����Ϊ�����ࡣ����һ�𣬲���֮ǰ���ڵ�����(��д����ͬʱ����һ������)
 * </table>
 *
 ***********************************************************************************
 * @par �޸���־:
 * <table>
 * <tr><th>Date                 <th>Version    <th>Author     <th>Description
 * <tr><td>2021��1��11��15:27:52<td>1.0.0.3    <td>naonao     <td>�����˽�YAML�ļ�תΪdat�ļ����ࡣΪ�˻�ͼ����
 * </table>
 *
 ***********************************************************************************
 * @par �޸���־:
 * <table>
 * <tr><th>Date                 <th>Version    <th>Author     <th>Description
 * <tr><td>2021��1��12��09:25:19<td>1.0.0.4    <td>naonao     <td>��д��ʱ��ĺ����޸�Ϊ��_s�ĺ����������˾��棬����������������ʹ��ʱ����cpp�ļ������������.h�ļ��а��������LLINK2005�ض������
 * </table>
 *
 * **********************************************************************************
 * @par �޸���־:
 * <table>
 * <tr><th>Date                 <th>Version    <th>Author     <th>Description
 * <tr><td>2021��1��14��09:55:37<td>1.0.0.5    <td>naonao     <td>���ֶ�ȡ����YAML�ļ���ȡ��ֹTab�����޸��ļ���ʱ���ÿո�������
 * </table>
 *
 * **********************************************************************************
 * @par �޸���־:
 * <table>
 * <tr><th>Date                 <th>Version    <th>Author     <th>Description
 * <tr><td>2021��1��22��09:57:34<td>1.0.0.6    <td>naonao     <td>����δ���ӽڵ�Ķ�д���������䣬�鿴��yml json xml�ĶԱȣ�����ѡ��xml��ʽ��Ϊ�ļ������塣���޸�ͳһ��ʽ��
 * </table>
 *
 * **********************************************************************************
 * @par �޸���־:
 * <table>
 * <tr><th>Date                 <th>Version    <th>Author     <th>Description
 * <tr><td>2021��1��25��10:58:27<td>2.0.0.1    <td>naonao     <td>���ӶԲ�ͬ�����ӽڵ��д�������������ʹ���ڵ��ڿ��԰�����ͬ�����͡�
 * </table>
 *
 * **********************************************************************************
 * @par �޸���־:
 * <table>
 * <tr><th>Date                 <th>Version    <th>Author     <th>Description
 * <tr><td>2021��1��27��09:17:12<td>2.0.0.2    <td>naonao     <td>�޸��˶�ʱ��д�룬��\n���޷�д������⣬�����޷��������������֮ǰ��ʱ���ʽ���˰汾���Կ����Լ�����ע������,
 * </table>
 *
 * **********************************************************************************
 * @par �޸���־:
 * <table>
 * <tr><th>Date                 <th>Version    <th>Author     <th>Description
 * <tr><td>2021��1��27��10:32:39<td>2.0.0.3    <td>naonao     <td>������dat�Ծ����д�룬δ��ʹ��mat�������ͣ���Ϊ��ά����ʵ�֣�֮�����������޸ġ�
 * </table>
 *
 * **********************************************************************************
 * @par �޸���־:
 * <table>
 * <tr><th>Date                 <th>Version    <th>Author     <th>Description
 * <tr><td> 2021��1��28��13:49:53<td>2.0.0.4   <td>naonao     <td>������dat�Ծ����д�룬ʵ����mat���͵�д�룬�����ڽ������ӡ�
 * </table>
 *
 * * **********************************************************************************
 * @par �޸���־:
 * <table>
 * <tr><th>Date                 <th>Version    <th>Author     <th>Description
 * <tr><td>2021��2��1��10:21:55 <td>2.0.0.4    <td>naonao     <td>�޸�ʧ�ܣ��汾���䡣Mat�Ĳ�ͬ����д���޸�ʧ��,�޸��˲��ִ���ṹ��ʹ֮����ࡣ
 * </table>
 * * **********************************************************************************
 * @par �޸���־:
 * <table>
 * <tr><th>Date                 <th>Version    <th>Author     <th>Description
 * <tr><td>2021/5/17 13:13:23   <td>2.0.0.5    <td>naonao     <td>���ض�ȡ�����ӽڵ�Ľڵ�ĺ�����ʹ֮���Զ�ȡǶ������
 * </table>
 * * **********************************************************************************
 * @par �޸���־:
 * <table>
 * <tr><th>Date                 <th>Version    <th>Author     <th>Description
 * <tr><td>2021/9/17 15:51:58   <td>2.0.0.6    <td>naonao     <td>�����޸�
 * </table>
 ***/
#pragma once
#ifndef __XMLWR_H__
#define __XMLWR_H__

#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <fstream>
#include <iostream>
#include <map>
#include <time.h>

#include "opencv2/opencv.hpp"
#define CV_VERSION_ID           \
    CVAUX_STR(CV_MAJOR_VERSION) \
    CVAUX_STR(CV_MINOR_VERSION) \
    CVAUX_STR(CV_SUBMINOR_VERSION)
// #ifdef _DEBUG
// #define cvLIB(name) "opencv_" name CV_VERSION_ID "d"
// #else
// #define cvLIB(name) "opencv_" name CV_VERSION_ID
// #endif
//
// #pragma comment( lib, cvLIB("img_hash") )
// #pragma comment( lib, cvLIB("world") )
namespace nao {
/**
 * @brief xml
 * @ref https://blog.csdn.net/zhaoyong26/article/details/84635383
 * @ref https://blog.csdn.net/ybhjx/article/details/50464025?utm_source=blogkpcl6
 * @ref https://blog.csdn.net/xingcen/article/details/55669054?utm_medium=distribute.pc_relevant.none-task-blog-baidujs_baidulandingword-2&spm=1001.2101.3001.4242
 * @ref https://blog.csdn.net/u010368556/article/details/79333503
 * @ref https://blog.csdn.net/dujian996099665/article/details/8879184
 * @ref https://blog.csdn.net/sss_369/article/details/92747709
 * @ref https://blog.csdn.net/iracer/article/details/51339377
 * @ref https://blog.csdn.net/owen7500/article/details/51029683
 * @ref https://blog.csdn.net/yaked/article/details/44301857
 * @ref https://blog.csdn.net/daniaokuye/article/details/78445753
 * @ref https://blog.csdn.net/xxyhjy/article/details/45485619
 * @ref https://blog.csdn.net/u013021895/article/details/52045410
 * @ref https://segmentfault.com/a/1190000021709051
 * @ref https://blog.csdn.net/xuejiren/article/details/25082765
 * @ref https://blog.csdn.net/qq_39534332/article/details/89784464
 * @ref https://blog.csdn.net/weixin_34194379/article/details/92288825
 * @ref https://blog.csdn.net/u011008379/article/details/18984703   �����������sstream��
 * @ref https://blog.csdn.net/xiongmingkang/article/details/83344456
 * @ref https://blog.csdn.net/weixin_34138377/article/details/94707551?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-1.control&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-1.control
 */
namespace xml {

    /**
     * @brief
     */
    class Xmlw {
    public:
        typedef std::map<std::string, int> mapsi;
        typedef std::map<std::string, int>::iterator mapsiit;
        explicit Xmlw(const int& type_w, const std::string& file_name);
        ~Xmlw();

    private:
        cv::FileStorage _fs;
        int _type_w;
        std::string _file_name;
        mapsi _parent_node;

    public:
        cv::FileStorage get_xml_write_object();

        template <typename T>
        void writeValue(const std::string& key, const T& value);
        template <typename T>
        void writeArray(const std::string& key, const std::vector<T>& value);

        void writeString(const std::string& key, const std::string& content);
        void writeString(const std::string& key, const std::vector<std::string>& contents);

        template <typename T>
        void writePoint(const std::string& key, const T& point);
        template <typename T>
        void writePoint(const std::string& key, const std::vector<T>& points);

        void writeMat(const std::string& key, const cv::Mat& matrix);
        void writeMat(const std::string& key, const std::vector<cv::Mat>& matrixs);

        template <typename T>
        void writeWithLeaf(const std::string& parentNodeName, const std::vector<std::string>& childNodeName, std::vector<T>& data);

        void startParentNode(const std::string& parentNodeName);
        template <typename T>
        void writeChildNode(const std::string& childNodeName, T& data);
        void endParentNode();
    }; // class xmlw

    class Xmlr {
    public:
        explicit Xmlr(const std::string& file_name);
        ~Xmlr();

    private:
        std::string _file_name;

    public:
        template <typename T>
        void readValue(const std::string& key, T& value);
        template <typename T>
        void readArray(const std::string& key, std::vector<T>& value);

        void readString(const std::string& key, std::string& content);
        void readString(const std::string& key, std::vector<std::string>& contents);

        template <typename T>
        void readPoint(const std::string& key, T& point);
        template <typename T>
        void readPoint(const std::string& key, std::vector<T>& points);

        void readMat(const std::string& key, cv::Mat& matrix);
        void readMat(const std::string& key, std::vector<cv::Mat>& matrixs);

        template <typename T>
        void readWithLeaf(const std::string& parentNodeName, const std::vector<std::string>& childNodeName, std::vector<T>& data);

        template <typename T>
        void readChildNode(const std::string& parentNodeName, const std::string& childNodeName, T& data);

        template <typename T>
        void readChildNode(const std::vector<std::string>& parentNodeName, const std::string& childNodeName, T& data);
    }; // class xmlr

    class Xml2Dat {
    public:
        explicit Xml2Dat(const std::string& filename);
        ~Xml2Dat();

    private:
        std::string _file_name;
        std::ofstream _os;

    public:
        template <typename T>
        void writeValue(const std::string& key, T& value);
        template <typename T>
        void writePoint(const std::string& key, std::vector<T>& points);
        template <typename T>
        void writeMat(const std::string& key, std::vector<std::vector<T>>& matrixs);
        template <typename T>
        void writeMat(const std::string& key, T& matrix);
    }; // class Xml2Dat
} // namespace xml
} // namespace nao
#include "xml_wr-inl.h"
#endif //__XMLWR_H__
/*----------------------------------------------------------------------------- (C) COPYRIGHT LEI *****END OF FILE------------------------------------------------------------------------------*/