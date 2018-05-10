template<class T>
void release_ptr(T* ptr) {
    if (nullptr != ptr) {
        delete ptr;
        ptr = nullptr;
    }
}

template<class T>
void release_array(T* ptr) {
    if (nullptr != ptr) {
        delete[] ptr;
        ptr = nullptr;
    }
}