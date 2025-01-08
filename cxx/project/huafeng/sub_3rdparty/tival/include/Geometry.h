#pragma once
#include "CommonDefine.h"


namespace Tival
{
    enum ExportAPI ShapeType
    {
        stUnknown = 0,
        stPoint,
        stLine,
        stRect,
        stRotRect,
        stCircle,
        stPolygon,
        stArrow,
        stText,
    };


    class ExportAPI TShapeBase
    {
    public:
        TShapeBase() {};
        virtual ~TShapeBase() {};
        virtual std::vector<double> Points() const = 0;
        virtual ShapeType Type() const = 0;
        virtual bool IsEmpty() { return _IsEmpty;}
        void SetEmpty(bool empty) { _IsEmpty = empty; }

        template<typename T>
        static T Empty() {
            T obj;
            obj.SetEmpty(true);
            return obj;
        }
    protected:
        bool _IsEmpty = false;
    };


    class ExportAPI TPoint : public TShapeBase
    {
    public:
        TPoint(): X(0), Y(0) {}
        TPoint(double x, double y):X(x),Y(y) {};
        virtual std::vector<double> Points() const { return {X, Y}; }
        virtual ShapeType Type() const { return ShapeType::stPoint; };

        double Distance(const TPoint& p) const;
        double Equal(const TPoint& p) const { return X == p.X && Y==p.Y; };
        bool operator==(const TPoint& p) const { return Equal(p); }
        static double Distance(const TPoint& p1, const TPoint& p2);
    protected:
        TPoint(bool empty) { _IsEmpty = empty; }
        
    public:
        double X;
        double Y;
    };

    class ExportAPI TLine
    {
    public:
        TLine(): X1(0),Y1(0),X2(0),Y2(0) {}
        TLine(double x1, double y1, double x2, double y2):X1(x1),Y1(y1),X2(x2),Y2(y2) {};
        TLine(const TPoint& start, const TPoint& end): X1(start.X),Y1(start.Y),X2(end.X),Y2(end.Y) {}

        virtual std::vector<double> Points() const { return {X1, Y1, X2, Y2}; }
        virtual ShapeType Type() { return ShapeType::stLine; };
        
        void Distance(const TLine& line, double& distMin, double& distMax) const;
        double VectorAngle() const;
        double VectorRadian() const;
        double LineLength() const;
        TPoint GetProjectPoint(TPoint p) const;
        TPoint GetCenterPoint() const;
        bool IsEmpty() const;

    public:
        double X1;
        double Y1;
        double X2;
        double Y2;
    };


    class ExportAPI TRect
    {
    public:
        TRect(): X1(0),Y1(0),X2(0),Y2(0) {}
        TRect(double x1, double y1, double x2, double y2):X1(x1),Y1(y1),X2(x2),Y2(y2) {};
        TRect(const json& rectParam);

        virtual std::vector<double> Points() const { return {X1, Y1, X2, Y2}; }
        virtual ShapeType Type() const { return ShapeType::stRect; };

        static TRect FromPtSize(double x, double y, double width, double height)
        {
            return TRect(x, y, x+width, y+height);
        }
    public:
        double X1;
        double Y1;
        double X2;
        double Y2;
    };

    class ExportAPI TRotateRect
    {
    public:
         double X;
         double Y;
         double Angle;
         double Length1;
         double Length2;
        TRotateRect():X(0),Y(0),Angle(0),Length1(0),Length2(0) {}
        TRotateRect(double y, double x, double angle, double length1, double length2)
        : X(x),Y(y),Angle(angle),Length1(length1),Length2(length2)
        {
        }

        /**
         * X = params["OriginX"];
         * Y = params["OriginY"];
         * Angle = params["RectAngle"];
         * Length1 = params["SizeX"];
         * Length2 = params["SizeY"];
        */
        TRotateRect(const json& params);
        virtual ShapeType Type() const { return ShapeType::stRotRect; };
        void GetCoords(double* xCoords, double* yCoords) const;
        TLine ToLine() const;
    };

    class ExportAPI TCircle
    {
    public:
        TCircle(): X(0),Y(0),Radius(0) {}
        TCircle(double x, double y, double radius, double start_phi=0, double end_phi = PI*2):X(x),Y(y),Radius(radius),StartPhi(start_phi),EndPhi(end_phi) {};
        virtual ShapeType Type() const { return ShapeType::stCircle; };

    public:
        double X;
        double Y;
        double Radius;
        double StartPhi = 0;
        double EndPhi = PI * 2;
    };
}
