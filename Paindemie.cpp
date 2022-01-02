// Simulation Parameters
#define PARAMS_POPULATION_SIZE   250000
#define PARAMS_INITIAL_INCIDENCE 5.0 / PARAMS_POPULATION_SIZE
#define PARAMS_PERSON_SPEED      0.01
#define PARAMS_RECOVER_CYCLES    15

#define NOMINMAX

#include <SDL.h>

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <CL/cl.hpp>

std::default_random_engine & GetRandomness()
{
    static std::default_random_engine randomness;
    return randomness;
}

template <typename T>
inline T Sq(T x)
{
    return x * x;
}

class Vec2
{
public:
    double x_;
    double y_;

    Vec2(double x, double y) : x_(x), y_(y)
    {
    }

#define IMPL_OP(op)                                                                                                                                            \
    IMPL_CMP(op) Vec2 & operator op##=(Vec2 const & right)                                                                                                     \
    {                                                                                                                                                          \
        x_ op## = right.x_;                                                                                                                                    \
        y_ op## = right.y_;                                                                                                                                    \
        return *this;                                                                                                                                          \
    }
#define IMPL_CMP(op)                                                                                                                                           \
    friend Vec2 operator op(Vec2 const & left, Vec2 const & right)                                                                                             \
    {                                                                                                                                                          \
        return Vec2(left.x_ op right.x_, left.y_ op right.y_);                                                                                                 \
    }

    IMPL_OP(+)
    IMPL_OP(-)
    IMPL_OP(/)
    IMPL_OP(*)

    friend double SqNorm(Vec2 const & vec)
    {
        return Sq(vec.x_) + Sq(vec.y_);
    }

    friend double Norm(Vec2 const & vec)
    {
        return std::sqrt(SqNorm(vec));
    }

    friend Vec2 CrossProd(Vec2 const & left, Vec2 const & right)
    {
        return Vec2(left.x_ * right.y_ - left.y_ * right.x_, left.y_ * right.x_ - left.x_ * right.y_);
    }

    friend double CrossProd0(Vec2 const & left, Vec2 const & right)
    {
        return left.x_ * right.y_ - left.y_ * right.x_;
    }

    friend double DotProd(Vec2 const & left, Vec2 const & right)
    {
        return left.x_ * right.x_ + left.y_ * right.y_;
    }
};

// Checks if (x0,y0) - (x1,y1)
template <typename TYPE>
bool IsCrossing(TYPE x0, TYPE y0, TYPE x1, TYPE y1, TYPE x2, TYPE y2, TYPE x3, TYPE y3)
{
    auto const eps(std::numeric_limits<TYPE>::epsilon());

    Vec2 const p(x0, y0);
    Vec2 const r(x1 - x0, y1 - y0);
    Vec2 const q(x2, y2);
    Vec2 const s(x3 - x2, y3 - y2);

    if (std::abs(CrossProd0(r, s)) < eps)
    {
        if (std::abs(CrossProd0(q - r, s)) < eps)
        {
            // Colinear case
            auto t0(DotProd(q - p, r) / DotProd(r, r));
            auto t1(DotProd(q + s - p, r) / DotProd(r, r));

            if (DotProd(s, r) < 0)
            {
                std::swap(t0, t1);
            }

            return std::min(1.0, t1) - std::max(0.0, t0) >= 0;
        }
        else
        {
            return false;
        }
    }
    else
    {
        auto const t(CrossProd0(q - p, s) / CrossProd0(r, s));
        auto const u(CrossProd0(q - p, r) / CrossProd0(r, s));

        return t >= 0 && t <= 1 && u >= 0 && u <= 1;
    }

    return false;
}

template <typename TYPE>
static std::string TemplatizeCode(std::string const & typeString, std::string const & templateCode)
{
    std::string code(templateCode);
    std::string type(typeString);

    if (std::is_same<TYPE, float>::value)
    {
        type = "float";
    }
    else if (std::is_same<TYPE, double>::value)
    {
        type = "double";
    }

    size_t pos {};
    size_t const typeSize(typeString.size());
    while ((pos = code.find(typeString)) != std::string::npos)
    {
        code.replace(pos, typeSize, type);
    }

    return code;
}

template <typename TYPE>
static std::string OpenCLCode_IsCrossing()
{
    return TemplatizeCode<TYPE>("TYPE", R"OpenCL(

inline TYPE CrossProd0(TYPE2 a, TYPE2 b)
{
    return a.x * b.y - a.y * b.x;
}

inline bool IsCrossing(TYPE2 m0, TYPE2 m1, TYPE2 m2, TYPE2 m3)
{
    TYPE2 const p = m0;
    TYPE2 const r = m1 - m0;
    TYPE2 const q = m2;
    TYPE2 const s = m3 - m2;

    if (fabs(CrossProd0(r, s)) < FLT_EPSILON)
    {
        if (fabs(CrossProd0(q - r, s)) < FLT_EPSILON)
        {
            // Colinear case
            TYPE t0 = dot(q - p, r) / dot(r, r);
            TYPE t1 = dot(q + s - p, r) / dot(r, r);

            if (dot(s, r) < 0)
            {
                float tmp = t0;
                t0 = t1;
                t1 = tmp;
            }

            return min((TYPE)1, t1) - max((TYPE)0, t0) >= 0;
        }
        else
        {
            return false;
        }
    }
    else
    {
        TYPE const t = CrossProd0(q - p, s) / CrossProd0(r, s);
        TYPE const u = CrossProd0(q - p, r) / CrossProd0(r, s);

        return t >= 0.0F && t <= 1.0F && u >= 0.0F && u <= 1.0F;
    }

    return false;
}

)OpenCL");
}

template <typename TYPE = float>
class Person
{
public:
    TYPE x = std::uniform_real_distribution<>()(GetRandomness());
    TYPE y = std::uniform_real_distribution<>()(GetRandomness());
    TYPE previousX = x;
    TYPE previousY = y;

    TYPE vx = std::uniform_real_distribution<>(-PARAMS_PERSON_SPEED, PARAMS_PERSON_SPEED)(GetRandomness());
    TYPE vy = std::uniform_real_distribution<>(-PARAMS_PERSON_SPEED, PARAMS_PERSON_SPEED)(GetRandomness());

    enum
    {
        Neg,
        Pos
    } status = std::uniform_real_distribution<>()(GetRandomness()) > PARAMS_INITIAL_INCIDENCE ? Neg : Pos;

    int recoverIn = -1;
    int newPos = 0;
    int immune = 0;

    void Heartbeat()
    {
        // Dirty border conditions
        while (x > 1.0)
            x -= 1.0;
        while (x < 0.0)
            x += 1.0;
        while (y > 1.0)
            y -= 1.0;
        while (y < 0.0)
            y += 1.0;

        previousX = x;
        previousY = y;

        x += vx;
        y += vy;

        if (recoverIn == -1 && status == Pos)
        {
            recoverIn = PARAMS_RECOVER_CYCLES;
        }

        if (status == Pos)
        {
            recoverIn--;
        }

        if (recoverIn == -1 && status == Pos)
        {
            status = Neg;
            immune = true;
        }

        if (newPos)
        {
            newPos = false;
        }
    }

    void Interact(Person & other)
    {
        // auto const dist(sq(x - other.x) + sq(y - other.y));

        if (IsCrossing(previousX, previousY, x, y, other.previousX, other.previousY, other.x, other.y))
        // if (dist < sq(0.01))
        {
            if ((other.status == Pos && other.newPos == false) || (status == Pos && newPos == false))
            {
                if (!immune && status != Pos)
                {
                    status = Pos;
                    newPos = true;
                }

                if (!other.immune && other.status != Pos)
                {
                    other.status = Pos;
                    other.newPos = true;
                }
            }
        }
    }

    static const std::string OpenCLCode()
    {
        return TemplatizeCode<TYPE>("TYPE", R"OpenCL(

typedef struct
{
    TYPE2 current;
    TYPE2 previous;
    TYPE2 speed;
    int status;
    int recoverIn;
    int newPos;
    int immune;

} Person;

enum { Neg = 0, Pos = 1 };

inline void Heartbeat(global Person * person, int recoverCycle)
{
    // Dirty border conditions
    while (person->current.x > 1.0)
        person->current.x -= 1.0;
    while (person->current.x < 0.0)
        person->current.x += 1.0;
    while (person->current.y > 1.0)
        person->current.y -= 1.0;
    while (person->current.y < 0.0)
        person->current.y += 1.0;

    person->previous = person->current;
    person->current += person->speed;

    if (person->recoverIn == -1 && person->status == Pos)
    {
        person->recoverIn = (int)recoverCycle;
    }

    if (person->status == Pos)
    {
        person->recoverIn--;
    }

    if (person->recoverIn == -1 && person->status == Pos)
    {
        person->status = Neg;
        person->immune = 1;
    }

    if (person->newPos)
    {
        person->newPos = 0;
    }
}

void Interact(global Person * person, global Person * other)
{
    if (IsCrossing(person->previous, person->current, other->previous, other->current))
    {
        if ((other->status == Pos && !other->newPos) || (person->status == Pos && !person->newPos))
        {
            if (!person->immune && person->status != Pos)
            {
                person->status = Pos;
                person->newPos = 1;
            }

            if (!other->immune && other->status != Pos)
            {
                other->status = Pos;
                other->newPos = 1;
            }
        }
    }
}

)OpenCL");
    }
};

template <typename TYPE = float>
class Env
{
    std::vector<Person<TYPE>> population = std::vector<Person<>>(PARAMS_POPULATION_SIZE);

public:
    std::vector<Person<TYPE>> & Population()
    {
        return population;
    }

    bool Stats(std::ostream & os, int t)
    {
        auto positives(0);
        auto negatives(0);

        for (auto const & person : population)
        {
            positives += (person.status == Person<TYPE>::Pos);
            negatives += (person.status == Person<TYPE>::Neg);
        }

        os << t << "," << positives << "," << negatives << std::endl;

        return positives > 0;
    }

    void Draw(SDL_Surface * screenSurface)
    {
        auto const bgColor(SDL_MapRGB(screenSurface->format, 0xC0, 0xD0, 0xC0));
        SDL_FillRect(screenSurface, NULL, bgColor);

        if (screenSurface->format->BitsPerPixel == 32)
        {
            auto const fgColor(SDL_MapRGB(screenSurface->format, 0x0, 0x0, 0x0));
            auto const redColor(SDL_MapRGB(screenSurface->format, 0xFF, 0x0, 0x0));

            auto const width(screenSurface->w);
            auto const height(screenSurface->h);
            auto const pixels(static_cast<uint32_t *>(screenSurface->pixels));

            SDL_LockSurface(screenSurface);

            for (auto const & person : population)
            {
                auto const R(3);

                for (int dy = -R; dy <= R; ++dy)
                {
                    for (int dx = -R; dx <= R; ++dx)
                    {
                        if (Sq(dx) + Sq(dy) <= Sq(R))
                        {
                            auto const x(static_cast<int>(std::round(person.x * width + dx)));
                            auto const y(static_cast<int>(std::round(person.y * height + dy)));

                            if (x >= 0 && x < width && y >= 0 && y < height)
                            {
                                pixels[x + y * (screenSurface->pitch >> 2)] = (person.status == Person<TYPE>::Neg) ? fgColor : redColor;
                            }
                        }
                    }
                }
            }

            SDL_UnlockSurface(screenSurface);
        }
    }

    void Step()
    {
#pragma omp parallel for schedule(dynamic)
        for (intptr_t i = 0; i < static_cast<intptr_t>(population.size()); ++i)
        {
            population[i].Heartbeat();
        }

#pragma omp parallel for schedule(dynamic)
        for (intptr_t i = 0; i < static_cast<intptr_t>(population.size()); ++i)
        {
            auto & person1(population[i]);
            if (person1.status == Person<TYPE>::Pos)
                for (auto & person2 : population)
                {
                    person1.Interact(person2);
                }
        }
    }

    static std::string OpenCLCode()
    {
        return TemplatizeCode<TYPE>("TYPE", R"OpenCL(

kernel void HeartbeatAll(global Person * population, int recoverCycle, global int * stats)
{
    int const i = get_global_id(0);
    if (population[i].status == Pos)
    {
        atomic_add(stats, 1);
    }
    Heartbeat(population + i, recoverCycle);
}

kernel void InteractAll(global Person * population)
{
    int const i = get_global_id(0);
    int const populationSize = get_global_size(0);

    if (population[i].status == Pos)
    {
        for (int j = 0; j < populationSize; ++j)
        {
            Interact(population + i, population + j);
        }
    }
}

)OpenCL");
    }
};

template <typename TYPE = float>
class CLComputer
{
    cl::Program program;
    cl::Buffer populationBuffer;
    cl::Buffer statsBuffer;
    cl::Kernel heartbeatAllKernel;
    cl::Kernel interactAllKernel;
    int stats {};

public:
    CLComputer(Env<TYPE> & env)
        : program(OpenCLCode_IsCrossing<TYPE>() + Person<TYPE>::OpenCLCode() + Env<TYPE>::OpenCLCode(), true),
          populationBuffer(CL_MEM_READ_WRITE, env.Population().size() * sizeof(Person<TYPE>)), statsBuffer(CL_MEM_READ_WRITE, sizeof(int)),
          heartbeatAllKernel(program, "HeartbeatAll"), interactAllKernel(program, "InteractAll")

    {
        std::cerr << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(cl::Device::getDefault()) << std::endl;

        heartbeatAllKernel.setArg(0, populationBuffer);
        heartbeatAllKernel.setArg(1, (int)PARAMS_RECOVER_CYCLES);
        heartbeatAllKernel.setArg(2, statsBuffer);

        interactAllKernel.setArg(0, populationBuffer);
    }

    void Run(Env<TYPE> & env, bool first)
    {
        auto const N(env.Population().size());
        cl::CommandQueue queue(cl::Context::getDefault(), cl::Device::getDefault());

        cl::Event evt;
        std::vector<cl::Event> waitEvts;

        int const zero {};
        queue.enqueueWriteBuffer(statsBuffer, false, 0, sizeof(int), &zero, nullptr, &evt);

        waitEvts = { evt };
        if (first)
        {
            queue.enqueueWriteBuffer(populationBuffer, false, 0, N * sizeof(Person<TYPE>), env.Population().data(), &waitEvts, &evt);
            waitEvts = { evt };
        }

        queue.enqueueNDRangeKernel(heartbeatAllKernel, cl::NDRange::NDRange(0), cl::NDRange::NDRange(N), cl::NullRange, &waitEvts, &evt);

        waitEvts = { evt };
        queue.enqueueNDRangeKernel(interactAllKernel, cl::NDRange::NDRange(0), cl::NDRange::NDRange(N), cl::NullRange, &waitEvts, &evt);

        waitEvts = { evt };
        queue.enqueueReadBuffer(populationBuffer, false, 0, N * sizeof(Person<TYPE>), env.Population().data(), &waitEvts, &evt);

        waitEvts = { evt };
        queue.enqueueReadBuffer(statsBuffer, false, 0, sizeof(int), &stats, nullptr, &evt);

        queue.finish();
    }

    bool Stats(std::ostream & os, int t)
    {
        auto positives(stats);
        auto negatives(PARAMS_POPULATION_SIZE - stats);

        os << t << "," << positives << "," << negatives << std::endl;

        return positives > 0;
    }
};

int main(int argc, char ** argv)
{
    using Type = float;

    Env<Type> env;
    CLComputer<Type> clContext(env);

    int width = 1200;
    int height = 1200;

    SDL_Window * window = NULL;
    SDL_Surface * screenSurface = NULL;
    if (SDL_Init(SDL_INIT_VIDEO) < 0)
    {
        return 1;
    }
    window = SDL_CreateWindow("Pandemie", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, width, height, SDL_WINDOW_SHOWN);
    if (window == NULL)
    {
        return 1;
    }
    screenSurface = SDL_GetWindowSurface(window);
    std::ofstream output("data.csv");

    bool quit {};
    for (int t = 0; !quit; ++t)
    {
        // if (!env.Stats(output, t))
        //{
        //    break;
        //}
        clContext.Run(env, t == 0);
        if (!clContext.Stats(output, t))
        {
            break;
        }
        // env.Step();
        env.Draw(screenSurface);
        SDL_UpdateWindowSurface(window);

        SDL_Event e;
        while (SDL_PollEvent(&e) != 0)
        {
            // User requests quit
            if (e.type == SDL_QUIT)
            {
                quit = true;
            }
        }
    }

    SDL_DestroyWindow(window);
    SDL_Quit();

    system("python show.py");

    return 0;
}
