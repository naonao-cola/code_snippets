#pragma once
#ifndef __XML_WR_INL_H__
#define __XML_WR_INL_H__
#include "xml_wr.h"
namespace nao {
namespace xml {
    template <typename T>
    void Xmlw::writeValue(const std::string& key, const T& value)
    {
        _fs << key << value;
    }

    template <typename T>
    void Xmlw::writeArray(const std::string& key, const std::vector<T>& value)
    {
        _fs << key << "[";
        for (int i = 0; i < value.size(); i++)
            _fs << value[i];
        _fs << "]";
    }

    template <typename T>
    void Xmlw::writePoint(const std::string& key, const T& point)
    {
        _fs << key << point;
    }

    template <typename T>
    void Xmlw::writePoint(const std::string& key, const std::vector<T>& points)
    {
        _fs << key << "[" << points << "]";
    }

    template <typename T>
    void Xmlw::writeWithLeaf(const std::string& parentNodeName, const std::vector<std::string>& childNodeName,
        std::vector<T>& data)
    {
        if (childNodeName.size() != data.size())
            std::cerr << "" << std::endl;
        _fs << parentNodeName << "{";
        for (std::size_t i = 0; i < childNodeName.size(); i++)
            _fs << childNodeName[i] << data[i];
        _fs << "}";
    }

    template <typename T>
    void Xmlw::writeChildNode(const std::string& childNodeName, T& data)
    {
        _fs << childNodeName << data;
    }

    /*--------------------------------------------------------------------------------------------------
    --------------------------------------------------------------------------------------------------*/
    template <typename T>
    void Xmlr::readValue(const std::string& key, T& value)
    {
        cv::FileStorage fs(_file_name, cv::FileStorage::READ);
        value = T(fs[key]);
        fs.release();
    }

    template <typename T>
    void Xmlr::readArray(const std::string& key, std::vector<T>& value)
    {
        cv::FileStorage fs(_file_name, cv::FileStorage::READ);
        cv::FileNode arrayName = fs[key];
        cv::FileNodeIterator it = arrayName.begin(), it_end = arrayName.end();
        for (; it != it_end; it++) {
            value.push_back((T)(*it));
        }
        fs.release();
    }

    template <typename T>
    void Xmlr::readPoint(const std::string& key, T& point)
    {
        cv::FileStorage fs(_file_name, cv::FileStorage::READ);
        fs[key] >> point;
        fs.release();
    }

    template <typename T>
    void Xmlr::readPoint(const std::string& key, std::vector<T>& points)
    {
        cv::FileStorage fs(_file_name, cv::FileStorage::READ);
        fs[key][0] >> points;
        fs.release();
    }

    template <typename T>
    void Xmlr::readWithLeaf(const std::string& parentNodeName, const std::vector<std::string>& childNodeName, std::vector<T>& data)
    {
        cv::FileStorage fs(_file_name, cv::FileStorage::READ);
        cv::FileNode parentNode = fs[parentNodeName];
        cv::FileNodeIterator it = parentNode.begin(), it_end = parentNode.end(); // �˴�ָ����ָ���һ���ӽڵ�
        int i = 0;
        for (; it != it_end, i < childNodeName.size(); it++, i++) {
            T tmp;
            (*it) >> tmp;
            data.push_back(tmp);
        }
        fs.release();
    }

    template <typename T>
    void Xmlr::readChildNode(const std::string& parentNodeName, const std::string& childNodeName, T& data)
    {
        cv::FileStorage fs(_file_name, cv::FileStorage::READ);
        cv::FileNode tmpNode = fs[parentNodeName];
        if (tmpNode.isNone()) {
            std::cerr << "" << parentNodeName << "" << std::endl;
            fs.release();
        }
        tmpNode[childNodeName] >> data;
        fs.release();
    }

    template <typename T>
    void Xmlr::readChildNode(const std::vector<std::string>& parentNodeName, const std::string& childNodeName, T& data)
    {
        cv::FileStorage fs(_file_name, cv::FileStorage::READ);
        cv::FileNode tmpNode = fs[parentNodeName[0]];
        for (int i = 1; i < parentNodeName.size(); i++) {
            tmpNode = tmpNode[parentNodeName[i]];
            if (tmpNode.isNone()) {
                std::cerr << "" << parentNodeName[i] << "" << std::endl;
                fs.release();
            }
        }
        tmpNode[childNodeName] >> data;
        fs.release();
    }

    /*--------------------------------------------------------------------------------------------------
    --------------------------------------------------------------------------------------------------*/
    template <typename T>
    void Xml2Dat::writeValue(const std::string& key, T& value)
    {
        _os << std::setiosflags(std::ios::left) << std::setw(16) << key << value << std::endl;
    }

    template <typename T>
    void Xml2Dat::writePoint(const std::string& key, std::vector<T>& points)
    {
        _os << std::setiosflags(std::ios::left) << key << std::endl;
        for (std::size_t i = 0; i < points.size(); i++) {
            _os << std::setiosflags(std::ios::left) << std::setw(16) << points[i].x << points[i].y << std::endl;
        }
    }

    template <typename T>
    void Xml2Dat::writeMat(const std::string& key, std::vector<std::vector<T>>& matrixs)
    {
        _os << std::setiosflags(std::ios::left) << key << std::endl;
        for (std::size_t i = 0; i < matrixs.size(); i++) {
            for (std::size_t j = 0; j < matrixs[i].size(); j++) {
                _os << std::resetiosflags(std::ios::left) << std::setw(8) << matrixs[i][j];
            }
            _os << std::endl;
        }
    }

    template <typename T>
    void Xml2Dat::writeMat(const std::string& key, T& matrix)
    {
        _os << std::setiosflags(std::ios::left) << key << std::endl;
        cv::Mat tmp = matrix.clone();
        int type = tmp.type();
        int iValue = 0;
        float fValue = 0.0f;
        double dValue = 0.0;
        for (int i = 0; i < tmp.rows; i++) {
            for (int j = 0; j < tmp.cols; j++) {
                switch (type) {
                case CV_8U:
                    iValue = tmp.ptr<uchar>(i, j)[0];
                    _os << std::setprecision(16) << std::setw(64) << iValue;
                    break;
                case CV_32F:
                    fValue = tmp.ptr<float>(i, j)[0];
                    _os << std::setprecision(32) << std::setw(64) << fValue;
                    break;
                case CV_64F:
                    dValue = tmp.ptr<double>(i, j)[0];
                    _os << std::setprecision(64) << std::setw(64) << dValue;
                    break;
                default:
                    break;
                }
            }
            _os << std::endl;
        }
    }
} // namespace xml
} // namespace nao
#endif //__XML_WR_INL_H__
