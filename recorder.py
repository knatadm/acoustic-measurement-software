import time
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import chirp

# Импорт модуля анализа спектра
import analyzer as an


def waveforms(timestamps, wf_params, signal_type='sin'):
    """
    Математически генерирует массив отсчетов сигнала заданного типа
    (синусоида, свип-тон или белый шум) для заданных временных меток.
    """
    if signal_type == 'sin':
        # Генерация чистого тона (гармонического синусоидального сигнала)
        signal = np.sin(2 * np.pi * wf_params['freq'] * timestamps)
    elif signal_type == 'chirp':
        # Генерация свип-тона (сигнала с линейно изменяющейся частотой от f_start до f_end)
        signal = chirp(
            timestamps,
            f0=wf_params['f_start'],
            f1=wf_params['f_end'],
            t1=wf_params['duration'],
            method='linear',
            phi=-90
        )
    elif signal_type == 'white_noise':
        # Генерация белого шума (случайные значения со случайными частотами по нормальному закону)
        amplitude = wf_params.get('amplitude', 0.5)
        signal = amplitude * np.random.randn(len(timestamps))
    else:
        raise ValueError(f"Unknown signal type: {signal_type}")
    return signal


def generate_and_record(wf_params, signal_type='sin'):
    """
    Генерирует сигнал, воспроизводит его через динамики, записывает через микрофон,
    выполняет эту процедуру несколько циклов и усредняет полученные спектры.
    """
    # Создание временной сетки
    timestamps = np.arange(wf_params['sample_rate'] * wf_params['duration']) / wf_params['sample_rate']
    # Математическая генерация эталонного сигнала
    signal_generated = waveforms(timestamps, wf_params, signal_type=signal_type)

    signal_recorded = np.empty((wf_params['cycles'], len(timestamps)))
    len_fft = 2 ** (int(np.ceil(np.log2(len(timestamps)))) - 1)
    signal_amplitudes = np.empty((wf_params['cycles'], len_fft))
    signal_frequencies = np.empty((wf_params['cycles'], len_fft))

    # Цикл многократных измерений
    for cycle in range(wf_params['cycles']):
        # Одновременное воспроизведение сгенерированного звука и запись ответа с микрофона
        recording = sd.playrec(signal_generated,
                               samplerate=wf_params['sample_rate'],
                               channels=1)
        sd.wait()  # Блокировка программы до полного завершения записи текущего цикла
        signal_recorded[cycle, :] = recording[:, 0]  # Сохранение записанной дорожки

        # Расчет спектра для свежезаписанного цикла через модуль analyzer
        freqs, amps = an.signal_spectrum(signal_recorded[cycle, :],
                                         wf_params['sample_rate'])
        # Ограничение данных размером окна len_fft
        signal_frequencies[cycle, :] = freqs[:len_fft]
        signal_amplitudes[cycle, :] = amps[:len_fft]

        # Пауза между циклами измерений (поправка на эхо в помещение)
        if cycle < wf_params['cycles'] - 1:
            time.sleep(wf_params['cycles_pause'])

    # Математическое усреднение амплитуд по всем проведенным циклам
    signal_amplitudes_avg = np.mean(signal_amplitudes, axis=0)
    signal_frequencies_avg = signal_frequencies[0, :]

    # Возврат усредненного спектра, временных меток и массивов сигналов
    return (signal_frequencies_avg, signal_amplitudes_avg,
            timestamps, signal_recorded, signal_generated)


def discrete_sin(wf_params, freq_array, path2save):
    """
    Пошагово генерирует и записывает чистые синусоиды для каждой частоты из массива freq_array,
    извлекая амплитуду отклика и сохраняя результаты в текстовый файл (метод дискретного свипа).
    """
    amplitudes_sweep = []
    # Создание/перезапись файла результатов с заголовками столбцов
    with open(f'{path2save}/amplitudes_sweep.txt', 'w') as f:
        f.write("Frequency (Hz), Amplitude\n\n")

    # Итерация по всем заданным частотам из массива
    for idx, freq in enumerate(freq_array):
        wf_params['freq'] = freq  # Подстановка текущей частоты для генерации
        # Воспроизведение и запись одиночной синусоиды
        (freqs_avg, amps_avg, _,
         _, _) = generate_and_record(wf_params, signal_type='sin')

        # Поиск индекса частоты в спектре, ближайшего к тестируемой частоте
        idx_freq = np.argmin(np.abs(freqs_avg - freq))
        amp = amps_avg[idx_freq]  # Фиксация амплитуды отклика на этой частоте
        amplitudes_sweep.append(amp)

        # Дозапись новой строки с результатами в файл
        with open(f'{path2save}/amplitudes_sweep.txt', 'a') as f:
            f.write(f"{freq}, {amp}\n")

        # Вывод лога в консоль о прогрессе выполнения эксперимента
        print(f"Recorded: {freq} Hz. Remaining: {len(freq_array) - idx - 1} of {len(freq_array)}")
        # Пауза перед переходом к следующей частоте
        if idx < len(freq_array) - 1:
            time.sleep(wf_params.get('sweep_pause', 1))

    return amplitudes_sweep


def gr_workflow(wf_params, path2save):
    """
    Главный управляющий метод: выбирает сценарий работы в зависимости от параметров.
    """
    if wf_params['type'] == 'sweep':
        gr_workflow_sweep(wf_params, path2save)  # Запуск пошагового частотного сканирования
    else:
        gr_workflow_wf(wf_params, path2save)  # Запуск непрерывной генерации формы волны


def gr_workflow_sweep(wf_params, path2save):
    """
    Реализует сценарий пошагового свипа: формирует сетку частот, запускает измерение,
    строит результирующий график амплитудно-частотной характеристики (АЧХ) и сохраняет его.
    """
    # Создание массива тестируемых частот от f_start до f_end с шагом f_step
    freq_array = np.arange(wf_params['f_start'],
                           wf_params['f_end'] + wf_params['f_step'],
                           wf_params['f_step'])
    # Получение списка амплитуд откликов для каждой частоты
    amplitudes = discrete_sin(wf_params, freq_array, path2save)

    # Отрисовка графика АЧХ
    plt.figure()
    plt.plot(freq_array, amplitudes, 'b-', label='Amplitude')  # Синяя сплошная линия графика
    plt.plot(freq_array, amplitudes, 'r.', markersize=4)  # Красные маркеры точек измерений
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    if wf_params['f_end'] > wf_params['f_start']:
        plt.xlim([wf_params['f_start'], wf_params['f_end']])  # Масштабирование по осям
    plt.grid(True)  # Включение координатной сетки
    plt.savefig(f'{path2save}/amplitudes_sweep.png')  # Сохранение графика в PNG-картинку
    plt.close()


def gr_workflow_wf(wf_params, path2save):
    """
    Реализует сценарий одиночной генерации: записывает аудиофайлы (.wav) сгенерированного
    и записанных сигналов, а также экспортирует их спектры в текстовые файлы для отчетов.
    """
    # Генерация и проведение серии циклических записей
    (freqs_avg, amps_avg, timestamps,
     recorded_signals, generated_signal) = generate_and_record(
        wf_params, signal_type=wf_params['type']
    )

    # Экспорт сгенерированного эталонного сигнала в аудиофайл формата .wav
    wavfile.write(f"{path2save}/generated_signal.wav",
                  wf_params['sample_rate'],
                  generated_signal.astype(np.float32))

    # Расчет спектра эталона и сохранение его числовой матрицы частота/амплитуда в .txt
    gen_freq, gen_amp = an.signal_spectrum(generated_signal, wf_params['sample_rate'])
    np.savetxt(f"{path2save}/signal_generated.txt",
               np.c_[gen_freq, gen_amp],
               header="Frequency (Hz), Amplitude")

    # Поцикловое сохранение всех полученных аудиозаписей с микрофона в отдельные файлы .wav
    for cycle in range(wf_params['cycles']):
        wavfile.write(f"{path2save}/recorded_signal_{cycle}.wav",
                      wf_params['sample_rate'],
                      recorded_signals[cycle, :].astype(np.float32))

    # Сохранение итогового усредненного по циклам спектра отклика в текстовый файл
    np.savetxt(f"{path2save}/signal_recorded_avg.txt",
               np.c_[freqs_avg, amps_avg],
               header="Frequency (Hz), Amplitude")
